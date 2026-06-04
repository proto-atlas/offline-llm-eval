import asyncio
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from offline_llm_eval.dataset.repository import Dataset
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.run.heartbeat import (
    HEARTBEAT_INTERVAL_SECONDS,
    RunRecord,
    RunStatus,
    update_run_heartbeat,
)

OLD_HEARTBEAT = datetime(2026, 5, 26, 8, 0, 0)
NEW_HEARTBEAT = datetime(2026, 5, 26, 8, 5, 0)


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def migrate_database(database_path: Path) -> None:
    command.upgrade(make_config(database_path), "head")


def create_test_engine(database_path: Path) -> AsyncEngine:
    return create_async_db_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")


async def create_run(
    session: AsyncSession,
    *,
    status: RunStatus = RunStatus.RUNNING,
    last_heartbeat_at: datetime = OLD_HEARTBEAT,
) -> int:
    dataset = Dataset(name="heartbeat_dataset", dataset_version="1.0.0")
    session.add(dataset)
    await session.flush()
    run = RunRecord(
        dataset_id=dataset.dataset_id,
        target_label="local",
        target_version="test",
        status=status.value,
        started_at=OLD_HEARTBEAT - timedelta(minutes=1),
        last_heartbeat_at=last_heartbeat_at,
    )
    session.add(run)
    await session.flush()
    return run.run_id


def fetch_last_heartbeat(database_path: Path) -> str:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("select last_heartbeat_at from runs").fetchone()

    assert row is not None
    assert isinstance(row[0], str)
    return row[0]


def test_run_insert_sets_last_heartbeat_at_default(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    asyncio.run(run_insert_sets_last_heartbeat_at_default(database_path))


async def run_insert_sets_last_heartbeat_at_default(database_path: Path) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="default_heartbeat", dataset_version="1.0.0")
            session.add(dataset)
            await session.flush()
            run = RunRecord(
                dataset_id=dataset.dataset_id,
                target_label="local",
                target_version=None,
                status=RunStatus.RUNNING.value,
            )
            session.add(run)

        async with session_factory() as session:
            result = await session.execute(select(RunRecord.last_heartbeat_at))

        assert result.scalar_one() is not None
    finally:
        await engine.dispose()


def test_update_run_heartbeat_commits_running_run(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    updated = asyncio.run(run_update_run_heartbeat_commits_running_run(database_path))

    assert updated is True
    assert fetch_last_heartbeat(database_path).startswith("2026-05-26 08:05:00")


async def run_update_run_heartbeat_commits_running_run(database_path: Path) -> bool:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            run_id = await create_run(session)

        return await update_run_heartbeat(session_factory, run_id, now=NEW_HEARTBEAT)
    finally:
        await engine.dispose()


def test_update_run_heartbeat_ignores_completed_run(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    updated = asyncio.run(run_update_run_heartbeat_ignores_completed_run(database_path))

    assert updated is False
    assert fetch_last_heartbeat(database_path).startswith("2026-05-26 08:00:00")


async def run_update_run_heartbeat_ignores_completed_run(database_path: Path) -> bool:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            run_id = await create_run(session, status=RunStatus.COMPLETED)

        return await update_run_heartbeat(session_factory, run_id, now=NEW_HEARTBEAT)
    finally:
        await engine.dispose()


def test_heartbeat_interval_is_thirty_seconds() -> None:
    assert HEARTBEAT_INTERVAL_SECONDS == 30.0
