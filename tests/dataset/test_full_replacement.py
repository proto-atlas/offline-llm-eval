import asyncio
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine

from offline_llm_eval.dataset.full_replacement import (
    DatasetImportBlockedByRunningRunError,
    import_dataset_full_replacement,
)
from offline_llm_eval.dataset.repository import Dataset
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus

NOW = datetime(2026, 5, 26, 12, 0, 0)
STALE_HEARTBEAT = datetime(2026, 5, 26, 10, 30, 0)
FRESH_HEARTBEAT = datetime(2026, 5, 26, 11, 30, 0)


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def migrate_database(database_path: Path) -> None:
    command.upgrade(make_config(database_path), "head")


def create_test_engine(database_path: Path) -> AsyncEngine:
    return create_async_db_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")


def dataset_document(
    *cases: dict[str, object],
    name: str = "replacement_dataset",
    version: str = "1.0.0",
) -> dict[str, object]:
    return {
        "name": name,
        "dataset_version": version,
        "cases": list(cases),
    }


def case_document(
    case_key: str,
    *assertions: dict[str, object],
    is_active: bool | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "case_key": case_key,
        "question": f"{case_key} question?",
        "assertions": list(assertions),
    }
    if is_active is not None:
        document["is_active"] = is_active
    return document


def assertion_document(assertion_id: str, *, is_active: bool | None = None) -> dict[str, object]:
    document: dict[str, object] = {
        "id": assertion_id,
        "type": "exact_match",
        "expected": assertion_id,
    }
    if is_active is not None:
        document["is_active"] = is_active
    return document


async def run_full_replacement(
    database_path: Path,
    document: dict[str, object],
) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            await import_dataset_full_replacement(session, document, now=NOW)
    finally:
        await engine.dispose()


def fetch_case_active(database_path: Path) -> dict[str, int]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "select case_key, is_active from evaluation_cases order by case_key"
        ).fetchall()
    return {row[0]: row[1] for row in rows}


def fetch_assertion_active(database_path: Path) -> dict[tuple[str, str], int]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            select evaluation_cases.case_key, assertions.id, assertions.is_active
            from assertions
            join evaluation_cases on evaluation_cases.case_id = assertions.case_id
            order by evaluation_cases.case_key, assertions.id
            """
        ).fetchall()
    return {(row[0], row[1]): row[2] for row in rows}


def fetch_run_statuses(database_path: Path) -> dict[int, str]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("select run_id, status from runs order by run_id").fetchall()
    return {row[0]: row[1] for row in rows}


def test_full_replacement_deactivates_removed_case_without_touching_assertions(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    asyncio.run(
        run_full_replacement(
            database_path,
            dataset_document(
                case_document("case_a", assertion_document("a1")),
                case_document("case_b", assertion_document("b1")),
            ),
        )
    )
    asyncio.run(
        run_full_replacement(
            database_path,
            dataset_document(case_document("case_a", assertion_document("a1"))),
        )
    )

    assert fetch_case_active(database_path) == {"case_a": 1, "case_b": 0}
    assert fetch_assertion_active(database_path)[("case_b", "b1")] == 1


def test_full_replacement_reactivates_reappeared_case_by_default(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    asyncio.run(
        run_full_replacement(
            database_path,
            dataset_document(
                case_document("case_a", assertion_document("a1")),
                case_document("case_b", assertion_document("b1")),
            ),
        )
    )
    asyncio.run(
        run_full_replacement(
            database_path,
            dataset_document(case_document("case_a", assertion_document("a1"))),
        )
    )
    asyncio.run(
        run_full_replacement(
            database_path,
            dataset_document(
                case_document("case_a", assertion_document("a1")),
                case_document("case_b", assertion_document("b1")),
            ),
        )
    )

    assert fetch_case_active(database_path) == {"case_a": 1, "case_b": 1}


def test_full_replacement_reconciles_assertions_for_active_case(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    asyncio.run(
        run_full_replacement(
            database_path,
            dataset_document(
                case_document("case_a", assertion_document("a1"), assertion_document("a2"))
            ),
        )
    )
    asyncio.run(
        run_full_replacement(
            database_path,
            dataset_document(
                case_document("case_a", assertion_document("a1"), assertion_document("a3"))
            ),
        )
    )

    assert fetch_assertion_active(database_path) == {
        ("case_a", "a1"): 1,
        ("case_a", "a2"): 0,
        ("case_a", "a3"): 1,
    }


def test_full_replacement_reactivates_reappeared_assertion_by_default(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    asyncio.run(
        run_full_replacement(
            database_path,
            dataset_document(
                case_document("case_a", assertion_document("a1"), assertion_document("a2"))
            ),
        )
    )
    asyncio.run(
        run_full_replacement(
            database_path,
            dataset_document(case_document("case_a", assertion_document("a1"))),
        )
    )
    asyncio.run(
        run_full_replacement(
            database_path,
            dataset_document(
                case_document("case_a", assertion_document("a1"), assertion_document("a2"))
            ),
        )
    )

    assert fetch_assertion_active(database_path)[("case_a", "a2")] == 1


def test_full_replacement_preserves_explicit_inactive_assertion(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    asyncio.run(
        run_full_replacement(
            database_path,
            dataset_document(
                case_document(
                    "case_a",
                    assertion_document("a1"),
                    assertion_document("a2", is_active=False),
                )
            ),
        )
    )

    assert fetch_assertion_active(database_path)[("case_a", "a2")] == 0


def test_full_replacement_sweeps_existing_dataset_before_import(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    asyncio.run(run_sweep_before_import_scenario(database_path))

    assert fetch_run_statuses(database_path) == {1: "aborted", 2: "running"}


async def run_sweep_before_import_scenario(database_path: Path) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            result = await import_dataset_full_replacement(
                session,
                dataset_document(case_document("case_a", assertion_document("a1"))),
                now=NOW,
            )
            other_dataset = Dataset(name="other_dataset", dataset_version="1.0.0")
            session.add(other_dataset)
            await session.flush()
            session.add(
                RunRecord(
                    dataset_id=result.dataset_id,
                    target_label="local",
                    target_version="test",
                    status=RunStatus.RUNNING.value,
                    started_at=STALE_HEARTBEAT - timedelta(hours=1),
                    last_heartbeat_at=STALE_HEARTBEAT,
                )
            )
            session.add(
                RunRecord(
                    dataset_id=other_dataset.dataset_id,
                    target_label="local",
                    target_version="test",
                    status=RunStatus.RUNNING.value,
                    started_at=STALE_HEARTBEAT - timedelta(hours=1),
                    last_heartbeat_at=STALE_HEARTBEAT,
                )
            )

        async with session_factory.begin() as session:
            await import_dataset_full_replacement(
                session,
                dataset_document(case_document("case_a", assertion_document("a1"))),
                now=NOW,
            )
    finally:
        await engine.dispose()


def test_full_replacement_rejects_active_running_run(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    with pytest.raises(DatasetImportBlockedByRunningRunError, match="実行中のrun 1"):
        asyncio.run(run_active_running_reject_scenario(database_path))


async def run_active_running_reject_scenario(database_path: Path) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            result = await import_dataset_full_replacement(
                session,
                dataset_document(case_document("case_a", assertion_document("a1"))),
                now=NOW,
            )
            session.add(
                RunRecord(
                    dataset_id=result.dataset_id,
                    target_label="local",
                    target_version="test",
                    status=RunStatus.RUNNING.value,
                    started_at=FRESH_HEARTBEAT - timedelta(minutes=5),
                    last_heartbeat_at=FRESH_HEARTBEAT,
                )
            )

        async with session_factory.begin() as session:
            await import_dataset_full_replacement(
                session,
                dataset_document(case_document("case_a", assertion_document("a2"))),
                now=NOW,
            )
    finally:
        await engine.dispose()


def test_full_replacement_skips_sweep_for_new_dataset(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    asyncio.run(run_new_dataset_skip_sweep_scenario(database_path))

    assert fetch_run_statuses(database_path) == {1: "running"}


async def run_new_dataset_skip_sweep_scenario(database_path: Path) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            other_dataset = Dataset(name="other_dataset", dataset_version="1.0.0")
            session.add(other_dataset)
            await session.flush()
            session.add(
                RunRecord(
                    dataset_id=other_dataset.dataset_id,
                    target_label="local",
                    target_version="test",
                    status=RunStatus.RUNNING.value,
                    started_at=STALE_HEARTBEAT - timedelta(hours=1),
                    last_heartbeat_at=STALE_HEARTBEAT,
                )
            )

        async with session_factory.begin() as session:
            await import_dataset_full_replacement(
                session,
                dataset_document(
                    case_document("case_a", assertion_document("a1")),
                    name="new_dataset",
                ),
                now=NOW,
            )
    finally:
        await engine.dispose()
