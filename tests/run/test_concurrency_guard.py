import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from offline_llm_eval.dataset.repository import Dataset
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.run.concurrency_guard import (
    CONCURRENT_RUN_BLOCKED_CODE,
    CONCURRENT_RUN_BLOCKED_EXIT_CODE,
    ConcurrentRunBlockedError,
    create_run_with_concurrency_guard,
    is_running_run_integrity_error,
)
from offline_llm_eval.run.heartbeat import RunStatus
from offline_llm_eval.run.repository import RunRepository

STARTED_AT = datetime(2026, 5, 27, 9, 0, 0)
COMPLETED_AT = datetime(2026, 5, 27, 9, 5, 0)


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def migrate_database(database_path: Path) -> None:
    command.upgrade(make_config(database_path), "head")


def create_test_engine(database_path: Path) -> AsyncEngine:
    return create_async_db_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")


async def create_dataset(
    session: AsyncSession,
    *,
    name: str = "concurrency_dataset",
    dataset_version: str = "1.0.0",
) -> int:
    dataset = Dataset(name=name, dataset_version=dataset_version)
    session.add(dataset)
    await session.flush()
    return dataset.dataset_id


def fetch_running_count(database_path: Path) -> int:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("select count(*) from runs where status = 'running'").fetchone()

    assert row is not None
    assert isinstance(row[0], int)
    return row[0]


def test_same_dataset_target_runningならconcurrent_run_blockedになる(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    error = asyncio.run(run_duplicate_running_run(database_path))

    assert error.code == CONCURRENT_RUN_BLOCKED_CODE
    assert error.exit_code == CONCURRENT_RUN_BLOCKED_EXIT_CODE
    assert error.dataset_id == 1
    assert error.target_label == "local"
    assert str(error) == "concurrent_run_blocked: dataset_id=1 target_label=local"


async def run_duplicate_running_run(database_path: Path) -> ConcurrentRunBlockedError:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset_id = await create_dataset(session)
            await create_run_with_concurrency_guard(
                RunRepository(session),
                dataset_id=dataset_id,
                target_label="local",
                started_at=STARTED_AT,
            )

        with pytest.raises(ConcurrentRunBlockedError) as exc_info:
            async with session_factory.begin() as session:
                await create_run_with_concurrency_guard(
                    RunRepository(session),
                    dataset_id=dataset_id,
                    target_label="local",
                    started_at=STARTED_AT,
                )

        return exc_info.value
    finally:
        await engine.dispose()


def test_same_dataset_targetでblockedされたrunは追加されない(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    asyncio.run(run_duplicate_running_run(database_path))

    assert fetch_running_count(database_path) == 1


def test_same_datasetでもtargetが違えばrunningを追加できる(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    run_ids = asyncio.run(run_different_target(database_path))

    assert run_ids == (1, 2)


async def run_different_target(database_path: Path) -> tuple[int, int]:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset_id = await create_dataset(session)
            first = await create_run_with_concurrency_guard(
                RunRepository(session),
                dataset_id=dataset_id,
                target_label="local",
                started_at=STARTED_AT,
            )
            second = await create_run_with_concurrency_guard(
                RunRepository(session),
                dataset_id=dataset_id,
                target_label="other",
                started_at=STARTED_AT,
            )

        return first.run_id, second.run_id
    finally:
        await engine.dispose()


def test_same_targetでもdatasetが違えばrunningを追加できる(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    run_ids = asyncio.run(run_different_dataset(database_path))

    assert run_ids == (1, 2)


async def run_different_dataset(database_path: Path) -> tuple[int, int]:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            first_dataset_id = await create_dataset(
                session,
                name="same_target_first",
                dataset_version="1.0.0",
            )
            second_dataset_id = await create_dataset(
                session,
                name="same_target_second",
                dataset_version="1.0.0",
            )
            first = await create_run_with_concurrency_guard(
                RunRepository(session),
                dataset_id=first_dataset_id,
                target_label="local",
                started_at=STARTED_AT,
            )
            second = await create_run_with_concurrency_guard(
                RunRepository(session),
                dataset_id=second_dataset_id,
                target_label="local",
                started_at=STARTED_AT,
            )

        return first.run_id, second.run_id
    finally:
        await engine.dispose()


def test_completedならsame_dataset_targetでもrunningを追加できる(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    snapshot_status = asyncio.run(run_after_completed(database_path))

    assert snapshot_status is RunStatus.RUNNING


async def run_after_completed(database_path: Path) -> RunStatus:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset_id = await create_dataset(session)
            created = await create_run_with_concurrency_guard(
                RunRepository(session),
                dataset_id=dataset_id,
                target_label="local",
                started_at=STARTED_AT,
            )
            await RunRepository(session).complete_run(created.run_id, completed_at=COMPLETED_AT)

        async with session_factory.begin() as session:
            second = await create_run_with_concurrency_guard(
                RunRepository(session),
                dataset_id=dataset_id,
                target_label="local",
                started_at=STARTED_AT,
            )

        return second.status
    finally:
        await engine.dispose()


def test_non_running_unique違反でなければ競合扱いしない() -> None:
    error = IntegrityError(
        "insert into runs (dataset_id, target_label, status) values (?, ?, ?)",
        (1, "local", None),
        RuntimeError("NOT NULL constraint failed: runs.status"),
    )

    assert is_running_run_integrity_error(error) is False
