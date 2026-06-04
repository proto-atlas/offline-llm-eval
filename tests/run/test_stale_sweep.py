import asyncio
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from offline_llm_eval.dataset.repository import Dataset
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus
from offline_llm_eval.run.stale_sweep import (
    DEFAULT_STALE_TIMEOUT,
    sweep_stale_runs_for_abort_stale,
    sweep_stale_runs_for_dataset_import,
    sweep_stale_runs_for_run_start,
)

NOW = datetime(2026, 5, 26, 12, 0, 0)
STALE_STARTED_AT = datetime(2026, 5, 26, 9, 0, 0)
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


async def create_dataset(session: AsyncSession, name: str) -> int:
    dataset = Dataset(name=name, dataset_version="1.0.0")
    session.add(dataset)
    await session.flush()
    return dataset.dataset_id


async def create_run(
    session: AsyncSession,
    *,
    dataset_id: int,
    target_label: str,
    status: RunStatus = RunStatus.RUNNING,
    heartbeat_at: datetime = STALE_HEARTBEAT,
) -> int:
    run = RunRecord(
        dataset_id=dataset_id,
        target_label=target_label,
        target_version="test",
        status=status.value,
        started_at=STALE_STARTED_AT,
        last_heartbeat_at=heartbeat_at,
    )
    session.add(run)
    await session.flush()
    return run.run_id


def fetch_statuses(database_path: Path) -> dict[int, str]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("select run_id, status from runs order by run_id").fetchall()

    return {row[0]: row[1] for row in rows}


def test_run_start_sweep_aborts_only_matching_dataset_and_target(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    aborted_run_ids = asyncio.run(run_start_sweep(database_path))

    assert aborted_run_ids == (1,)
    assert fetch_statuses(database_path) == {
        1: "aborted",
        2: "running",
        3: "running",
        4: "completed",
    }


async def run_start_sweep(database_path: Path) -> tuple[int, ...]:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            first_dataset_id = await create_dataset(session, "first")
            second_dataset_id = await create_dataset(session, "second")
            await create_run(session, dataset_id=first_dataset_id, target_label="local")
            await create_run(session, dataset_id=first_dataset_id, target_label="other")
            await create_run(session, dataset_id=second_dataset_id, target_label="local")
            await create_run(
                session,
                dataset_id=first_dataset_id,
                target_label="local",
                status=RunStatus.COMPLETED,
            )

        async with session_factory.begin() as session:
            result = await sweep_stale_runs_for_run_start(
                session,
                dataset_id=first_dataset_id,
                target_label="local",
                now=NOW,
            )

        return result.aborted_run_ids
    finally:
        await engine.dispose()


def test_dataset_import_sweep_aborts_dataset_across_targets(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    aborted_run_ids = asyncio.run(dataset_import_sweep(database_path))

    assert aborted_run_ids == (1, 2)
    assert fetch_statuses(database_path) == {
        1: "aborted",
        2: "aborted",
        3: "running",
    }


async def dataset_import_sweep(database_path: Path) -> tuple[int, ...]:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            first_dataset_id = await create_dataset(session, "first")
            second_dataset_id = await create_dataset(session, "second")
            await create_run(session, dataset_id=first_dataset_id, target_label="local")
            await create_run(session, dataset_id=first_dataset_id, target_label="other")
            await create_run(session, dataset_id=second_dataset_id, target_label="local")

        async with session_factory.begin() as session:
            result = await sweep_stale_runs_for_dataset_import(
                session,
                dataset_id=first_dataset_id,
                now=NOW,
            )

        return result.aborted_run_ids
    finally:
        await engine.dispose()


def test_abort_stale_sweep_aborts_all_scopes(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    aborted_run_ids = asyncio.run(abort_stale_sweep(database_path))

    assert aborted_run_ids == (1, 2)
    assert fetch_statuses(database_path) == {
        1: "aborted",
        2: "aborted",
        3: "running",
    }


async def abort_stale_sweep(database_path: Path) -> tuple[int, ...]:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            first_dataset_id = await create_dataset(session, "first")
            second_dataset_id = await create_dataset(session, "second")
            await create_run(session, dataset_id=first_dataset_id, target_label="local")
            await create_run(session, dataset_id=second_dataset_id, target_label="other")
            await create_run(
                session,
                dataset_id=second_dataset_id,
                target_label="fresh",
                heartbeat_at=FRESH_HEARTBEAT,
            )

        async with session_factory.begin() as session:
            result = await sweep_stale_runs_for_abort_stale(session, now=NOW)

        return result.aborted_run_ids
    finally:
        await engine.dispose()


def test_sweep_uses_started_at_when_legacy_heartbeat_is_null(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    create_legacy_runs_table(database_path)

    aborted_run_ids = asyncio.run(legacy_heartbeat_null_sweep(database_path))

    assert aborted_run_ids == (1,)
    assert fetch_statuses(database_path) == {1: "aborted"}


def create_legacy_runs_table(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            create table runs (
                run_id integer primary key,
                dataset_id integer not null,
                target_label varchar(255) not null,
                target_version varchar(255),
                status varchar(32) not null,
                started_at datetime not null,
                completed_at datetime,
                last_heartbeat_at datetime,
                gate_config_snapshot_json json,
                gate_result_json json,
                created_at datetime not null,
                updated_at datetime not null
            )
            """
        )
        connection.execute(
            """
            insert into runs (
                run_id,
                dataset_id,
                target_label,
                target_version,
                status,
                started_at,
                completed_at,
                last_heartbeat_at,
                created_at,
                updated_at
            ) values (
                1,
                1,
                'local',
                'test',
                'running',
                '2026-05-26 09:00:00.000000',
                null,
                null,
                '2026-05-26 09:00:00.000000',
                '2026-05-26 09:00:00.000000'
            )
            """
        )


async def legacy_heartbeat_null_sweep(database_path: Path) -> tuple[int, ...]:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            result = await sweep_stale_runs_for_abort_stale(session, now=NOW)

        return result.aborted_run_ids
    finally:
        await engine.dispose()


def test_default_stale_timeout_is_one_hour() -> None:
    assert DEFAULT_STALE_TIMEOUT == timedelta(hours=1)
