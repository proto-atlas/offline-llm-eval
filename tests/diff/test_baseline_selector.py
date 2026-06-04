import asyncio
from datetime import datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from offline_llm_eval.dataset.repository import Dataset
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.diff.baseline_selector import (
    BaselineInProgressError,
    BaselineNotFoundError,
    BaselineSelection,
    BaselineWarning,
    InvalidBaselineSpecError,
    select_baseline_run,
)
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus
from offline_llm_eval.run.repository import RunSnapshot

STARTED_AT = datetime(2026, 5, 27, 9, 0, 0)
COMPLETED_AT = datetime(2026, 5, 27, 9, 5, 0)
HEARTBEAT_AT = datetime(2026, 5, 27, 9, 4, 30)
OLD_CREATED_AT = datetime(2026, 5, 27, 8, 0, 0)
NEW_CREATED_AT = datetime(2026, 5, 27, 10, 0, 0)
CURRENT_CREATED_AT = datetime(2026, 5, 27, 11, 0, 0)


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def migrate_database(database_path: Path) -> None:
    command.upgrade(make_config(database_path), "head")


def create_test_engine(database_path: Path) -> AsyncEngine:
    return create_async_db_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")


def test_latestは同一dataset_targetのcompleted最新runを選ぶ(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    selection = asyncio.run(select_latest_baseline(database_path))

    assert selection.run.run_id == 2
    assert selection.run.status is RunStatus.COMPLETED
    assert selection.warnings == ()


async def select_latest_baseline(database_path: Path) -> BaselineSelection:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset_id = await create_dataset(session)
            await add_run(
                session,
                dataset_id=dataset_id,
                target_label="local",
                status=RunStatus.COMPLETED,
                created_at=OLD_CREATED_AT,
            )
            await add_run(
                session,
                dataset_id=dataset_id,
                target_label="local",
                status=RunStatus.COMPLETED,
                created_at=NEW_CREATED_AT,
            )
            await add_run(
                session,
                dataset_id=dataset_id,
                target_label="local",
                status=RunStatus.RUNNING,
                created_at=CURRENT_CREATED_AT,
            )
            current_run = snapshot(
                run_id=3,
                dataset_id=dataset_id,
                target_label="local",
                status=RunStatus.RUNNING,
            )
            return await select_baseline_run(
                session,
                current_run=current_run,
                baseline_spec="latest",
            )
    finally:
        await engine.dispose()


def test_latest_mainは同一datasetのmain最新completedを選ぶ(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    selection = asyncio.run(select_latest_main_baseline(database_path))

    assert selection.run.target_label == "main"
    assert selection.run.run_id == 2


async def select_latest_main_baseline(database_path: Path) -> BaselineSelection:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset_id = await create_dataset(session)
            await add_run(
                session,
                dataset_id=dataset_id,
                target_label="local",
                status=RunStatus.COMPLETED,
                created_at=NEW_CREATED_AT,
            )
            await add_run(
                session,
                dataset_id=dataset_id,
                target_label="main",
                status=RunStatus.COMPLETED,
                created_at=OLD_CREATED_AT,
            )
            current_run = snapshot(
                run_id=3,
                dataset_id=dataset_id,
                target_label="local",
                status=RunStatus.RUNNING,
            )
            return await select_baseline_run(
                session,
                current_run=current_run,
                baseline_spec="latest_main",
            )
    finally:
        await engine.dispose()


def test_run指定でrunningならbaseline_in_progressを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    with pytest.raises(BaselineInProgressError) as error:
        asyncio.run(select_running_explicit_baseline(database_path))

    assert error.value.code == "baseline_in_progress"
    assert error.value.exit_code == 1
    assert error.value.run_id == 1


async def select_running_explicit_baseline(database_path: Path) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset_id = await create_dataset(session)
            await add_run(
                session,
                dataset_id=dataset_id,
                target_label="local",
                status=RunStatus.RUNNING,
                created_at=OLD_CREATED_AT,
            )
            await select_baseline_run(
                session,
                current_run=snapshot(99, dataset_id, "local", RunStatus.RUNNING),
                baseline_spec="run:1",
            )
    finally:
        await engine.dispose()


def test_run指定でabortedならwarning付きで返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    selection = asyncio.run(select_aborted_explicit_baseline(database_path))

    assert selection.run.status is RunStatus.ABORTED
    assert selection.warnings == (BaselineWarning.BASELINE_ABORTED,)


async def select_aborted_explicit_baseline(database_path: Path) -> BaselineSelection:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset_id = await create_dataset(session)
            await add_run(
                session,
                dataset_id=dataset_id,
                target_label="local",
                status=RunStatus.ABORTED,
                created_at=OLD_CREATED_AT,
            )
            return await select_baseline_run(
                session,
                current_run=snapshot(99, dataset_id, "local", RunStatus.RUNNING),
                baseline_spec="run:1",
            )
    finally:
        await engine.dispose()


def test_latestが見つからない場合はbaseline_not_found(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    with pytest.raises(BaselineNotFoundError) as error:
        asyncio.run(select_missing_latest_baseline(database_path))

    assert error.value.code == "baseline_not_found"
    assert error.value.baseline_spec == "latest"


async def select_missing_latest_baseline(database_path: Path) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset_id = await create_dataset(session)
            await select_baseline_run(
                session,
                current_run=snapshot(1, dataset_id, "local", RunStatus.RUNNING),
                baseline_spec="latest",
            )
    finally:
        await engine.dispose()


def test不正なbaseline指定子はvalidation_error(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    with pytest.raises(InvalidBaselineSpecError) as error:
        asyncio.run(select_invalid_baseline_spec(database_path))

    assert error.value.code == "validation_error"
    assert error.value.baseline_spec == "run:abc"


async def select_invalid_baseline_spec(database_path: Path) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            await select_baseline_run(
                session,
                current_run=snapshot(1, 1, "local", RunStatus.RUNNING),
                baseline_spec="run:abc",
            )
    finally:
        await engine.dispose()


async def create_dataset(session: AsyncSession) -> int:
    dataset = Dataset(name="baseline_selector_dataset", dataset_version="1.0.0")
    session.add(dataset)
    await session.flush()
    return dataset.dataset_id


async def add_run(
    session: AsyncSession,
    *,
    dataset_id: int,
    target_label: str,
    status: RunStatus,
    created_at: datetime,
) -> None:
    run = RunRecord(
        dataset_id=dataset_id,
        target_label=target_label,
        target_version="mock-v1",
        status=status.value,
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT if status is not RunStatus.RUNNING else None,
        last_heartbeat_at=HEARTBEAT_AT,
        created_at=created_at,
        updated_at=created_at,
    )
    session.add(run)
    await session.flush()


def snapshot(
    run_id: int,
    dataset_id: int,
    target_label: str,
    status: RunStatus,
) -> RunSnapshot:
    return RunSnapshot(
        run_id=run_id,
        dataset_id=dataset_id,
        target_label=target_label,
        target_version="mock-v1",
        status=status,
        started_at=STARTED_AT,
        completed_at=None,
        last_heartbeat_at=HEARTBEAT_AT,
        gate_config_snapshot_json=None,
        gate_result_json=None,
    )
