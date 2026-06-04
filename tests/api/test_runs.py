import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from offline_llm_eval.api.runs import get_database_session
from offline_llm_eval.dataset.repository import Dataset
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.main import create_app
from offline_llm_eval.run.case_result import CaseResultRecord
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus
from offline_llm_eval.runner.error_type import ErrorType

STARTED_AT = datetime(2026, 5, 27, 10, 0, 0)
COMPLETED_AT = datetime(2026, 5, 27, 10, 5, 0)
HEARTBEAT_AT = datetime(2026, 5, 27, 10, 4, 30)
OLDER_CREATED_AT = datetime(2026, 5, 27, 9, 0, 0)
MIDDLE_CREATED_AT = datetime(2026, 5, 27, 9, 30, 0)
NEWER_CREATED_AT = datetime(2026, 5, 27, 11, 0, 0)


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def migrate_database(database_path: Path) -> None:
    command.upgrade(make_config(database_path), "head")


def create_test_engine(database_path: Path) -> AsyncEngine:
    return create_async_db_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")


def create_test_client(database_path: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_database_session] = make_session_override(database_path)
    return TestClient(app)


def make_session_override(database_path: Path) -> Callable[[], AsyncIterator[AsyncSession]]:
    async def override() -> AsyncIterator[AsyncSession]:
        engine = create_test_engine(database_path)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                yield session
        finally:
            await engine.dispose()

    return override


def test_run_summaryはmetadata_counts_rates_error_typeを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_completed_run_with_case_results(database_path))
    client = create_test_client(database_path)

    response = client.get(f"/api/runs/{run_id}")

    assert response.status_code == 200
    assert response.json() == {
        "run": {
            "run_id": 1,
            "dataset_id": 1,
            "dataset_name": "summary_dataset",
            "dataset_version": "1.0.0",
            "target_label": "local",
            "target_version": "mock-v1",
            "status": "completed",
            "started_at": "2026-05-27T10:00:00",
            "completed_at": "2026-05-27T10:05:00",
            "last_heartbeat_at": "2026-05-27T10:04:30",
        },
        "counts": {
            "total": 5,
            "executed": 4,
            "passed": 2,
            "failed": 1,
            "needs_review": 1,
            "skipped": 1,
        },
        "rates": {
            "pass_rate": 0.5,
            "fail_rate": 0.25,
            "skipped_ratio": 0.2,
        },
        "error_types": {
            "provider_error": 1,
            "overloaded": 1,
            "unknown_error": 0,
            "response_mode_mismatch": 1,
        },
        "usage": {
            "usage_source": "unavailable",
            "usage_json": None,
        },
        "gate": {
            "config_snapshot": {"pass_rate_min": 0.8},
            "result": {"verdict": "pass"},
        },
    }


def test_run_list(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_ids = asyncio.run(seed_runs_for_list(database_path))
    client = create_test_client(database_path)

    response = client.get("/api/runs", params={"limit": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 2
    assert body["target_label"] is None
    assert [run["run_id"] for run in body["runs"]] == [
        run_ids["newer_local"],
        run_ids["middle_remote"],
    ]
    assert body["runs"][0] == {
        "run_id": run_ids["newer_local"],
        "dataset_id": 1,
        "dataset_name": "list_dataset",
        "dataset_version": "1.0.0",
        "target_label": "local",
        "target_version": "mock-v2",
        "status": "aborted",
        "started_at": "2026-05-27T10:00:00",
        "completed_at": "2026-05-27T10:05:00",
        "last_heartbeat_at": "2026-05-27T10:04:30",
        "created_at": "2026-05-27T11:00:00",
    }


def test_run_listはtarget_labelで絞り込む(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_ids = asyncio.run(seed_runs_for_list(database_path))
    client = create_test_client(database_path)

    response = client.get("/api/runs", params={"target_label": "local"})

    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 100
    assert body["target_label"] == "local"
    assert [run["run_id"] for run in body["runs"]] == [
        run_ids["newer_local"],
        run_ids["older_local"],
    ]


def test_run_listはlimit上限超過なら422を返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    client = create_test_client(database_path)

    response = client.get("/api/runs", params={"limit": 1001})

    assert response.status_code == 422


async def seed_runs_for_list(database_path: Path) -> dict[str, int]:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="list_dataset", dataset_version="1.0.0")
            session.add(dataset)
            await session.flush()

            older_local = RunRecord(
                dataset_id=dataset.dataset_id,
                target_label="local",
                target_version="mock-v1",
                status=RunStatus.COMPLETED.value,
                started_at=STARTED_AT,
                completed_at=COMPLETED_AT,
                last_heartbeat_at=HEARTBEAT_AT,
                created_at=OLDER_CREATED_AT,
                updated_at=OLDER_CREATED_AT,
            )
            middle_remote = RunRecord(
                dataset_id=dataset.dataset_id,
                target_label="remote",
                target_version="mock-v1",
                status=RunStatus.RUNNING.value,
                started_at=STARTED_AT,
                last_heartbeat_at=HEARTBEAT_AT,
                created_at=MIDDLE_CREATED_AT,
                updated_at=MIDDLE_CREATED_AT,
            )
            newer_local = RunRecord(
                dataset_id=dataset.dataset_id,
                target_label="local",
                target_version="mock-v2",
                status=RunStatus.ABORTED.value,
                started_at=STARTED_AT,
                completed_at=COMPLETED_AT,
                last_heartbeat_at=HEARTBEAT_AT,
                created_at=NEWER_CREATED_AT,
                updated_at=NEWER_CREATED_AT,
            )
            session.add_all([older_local, middle_remote, newer_local])
            await session.flush()

            return {
                "older_local": older_local.run_id,
                "middle_remote": middle_remote.run_id,
                "newer_local": newer_local.run_id,
            }
    finally:
        await engine.dispose()


async def seed_completed_run_with_case_results(database_path: Path) -> int:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="summary_dataset", dataset_version="1.0.0")
            session.add(dataset)
            await session.flush()

            run = RunRecord(
                dataset_id=dataset.dataset_id,
                target_label="local",
                target_version="mock-v1",
                status=RunStatus.COMPLETED.value,
                started_at=STARTED_AT,
                completed_at=COMPLETED_AT,
                last_heartbeat_at=HEARTBEAT_AT,
                gate_config_snapshot_json={"pass_rate_min": 0.8},
                gate_result_json={"verdict": "pass"},
            )
            session.add(run)
            await session.flush()

            await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=run.run_id,
                case_key="case_pass_1",
                status=CaseResultStatus.PASS,
            )
            await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=run.run_id,
                case_key="case_pass_2",
                status=CaseResultStatus.PASS,
            )
            await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=run.run_id,
                case_key="case_failed",
                status=CaseResultStatus.FAILED,
                error_type=ErrorType.PROVIDER_ERROR,
            )
            await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=run.run_id,
                case_key="case_needs_review",
                status=CaseResultStatus.NEEDS_REVIEW,
                error_type=ErrorType.OVERLOADED,
            )
            await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=run.run_id,
                case_key="case_skipped",
                status=CaseResultStatus.SKIPPED,
                error_type=ErrorType.RESPONSE_MODE_MISMATCH,
            )

            return run.run_id
    finally:
        await engine.dispose()


async def add_case_result(
    session: AsyncSession,
    *,
    dataset_id: int,
    run_id: int,
    case_key: str,
    status: CaseResultStatus,
    error_type: ErrorType | None = None,
) -> None:
    await session.execute(
        text(
            """
            insert into evaluation_cases
                (dataset_id, case_key, question, severity, tags_json, metadata_json)
            values
                (:dataset_id, :case_key, 'Question?', 'medium', '[]', null)
            """
        ),
        {"dataset_id": dataset_id, "case_key": case_key},
    )
    result = await session.execute(
        text(
            """
            select case_id
            from evaluation_cases
            where dataset_id = :dataset_id and case_key = :case_key
            """
        ),
        {"dataset_id": dataset_id, "case_key": case_key},
    )
    case_id = result.scalar_one()
    session.add(
        CaseResultRecord(
            run_id=run_id,
            case_id=case_id,
            case_key=case_key,
            status=status.value,
            error_type=error_type.value if error_type is not None else None,
        )
    )


def test_run_summaryはcase_resultなしならzero_metricsを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_running_run_without_case_results(database_path))
    client = create_test_client(database_path)

    response = client.get(f"/api/runs/{run_id}")

    assert response.status_code == 200
    assert response.json()["counts"] == {
        "total": 0,
        "executed": 0,
        "passed": 0,
        "failed": 0,
        "needs_review": 0,
        "skipped": 0,
    }
    assert response.json()["rates"] == {
        "pass_rate": 0.0,
        "fail_rate": 0.0,
        "skipped_ratio": 0.0,
    }


async def seed_running_run_without_case_results(database_path: Path) -> int:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="empty_run_dataset", dataset_version="1.0.0")
            session.add(dataset)
            await session.flush()
            run = RunRecord(
                dataset_id=dataset.dataset_id,
                target_label="local",
                target_version=None,
                status=RunStatus.RUNNING.value,
                started_at=STARTED_AT,
                last_heartbeat_at=HEARTBEAT_AT,
            )
            session.add(run)
            await session.flush()

            return run.run_id
    finally:
        await engine.dispose()


def test_run_summaryはmissingならrun_not_foundを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    client = create_test_client(database_path)

    response = client.get("/api/runs/999")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "run_not_found",
            "message": "実行が見つかりません。",
            "extra": {"run_id": 999},
        },
    }
