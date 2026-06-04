import asyncio
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from offline_llm_eval.dataset.repository import Dataset
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.run.heartbeat import RunStatus
from offline_llm_eval.run.repository import RunRepository, RunSnapshot

STARTED_AT = datetime(2026, 5, 26, 9, 0, 0)
COMPLETED_AT = datetime(2026, 5, 26, 9, 5, 0)
ABORTED_AT = datetime(2026, 5, 26, 9, 10, 0)


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def migrate_database(database_path: Path) -> None:
    command.upgrade(make_config(database_path), "head")


def create_test_engine(database_path: Path) -> AsyncEngine:
    return create_async_db_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")


async def create_dataset(session: AsyncSession) -> int:
    dataset = Dataset(name="run_repository_dataset", dataset_version="1.0.0")
    session.add(dataset)
    await session.flush()
    return dataset.dataset_id


def test_create_run_running状態でgate_config_snapshotを保存する(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    snapshot = asyncio.run(run_create_run(database_path))

    assert snapshot.run_id == 1
    assert snapshot.status is RunStatus.RUNNING
    assert snapshot.target_label == "local"
    assert snapshot.target_version == "mock-v1"
    assert snapshot.started_at == STARTED_AT
    assert snapshot.completed_at is None
    assert snapshot.last_heartbeat_at is not None
    assert snapshot.gate_config_snapshot_json == {"pass_rate_min": 0.95}
    assert snapshot.gate_result_json is None


async def run_create_run(database_path: Path) -> RunSnapshot:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset_id = await create_dataset(session)
            return await RunRepository(session).create_run(
                dataset_id=dataset_id,
                target_label="local",
                target_version="mock-v1",
                gate_config_snapshot_json={"pass_rate_min": 0.95},
                started_at=STARTED_AT,
            )
    finally:
        await engine.dispose()


def test_complete_run_runningをcompletedにしてgate_resultを保存する(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    snapshot = asyncio.run(run_complete_run(database_path))

    assert snapshot is not None
    assert snapshot.status is RunStatus.COMPLETED
    assert snapshot.completed_at == COMPLETED_AT
    assert snapshot.gate_result_json == {"status": "pass"}


async def run_complete_run(database_path: Path) -> RunSnapshot | None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset_id = await create_dataset(session)
            created = await RunRepository(session).create_run(
                dataset_id=dataset_id,
                target_label="local",
                started_at=STARTED_AT,
            )

        async with session_factory.begin() as session:
            return await RunRepository(session).complete_run(
                created.run_id,
                completed_at=COMPLETED_AT,
                gate_result_json={"status": "pass"},
            )
    finally:
        await engine.dispose()


def test_abort_run_runningをabortedにする(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    snapshot = asyncio.run(run_abort_run(database_path))

    assert snapshot is not None
    assert snapshot.status is RunStatus.ABORTED
    assert snapshot.completed_at == ABORTED_AT
    assert snapshot.gate_result_json is None


async def run_abort_run(database_path: Path) -> RunSnapshot | None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset_id = await create_dataset(session)
            created = await RunRepository(session).create_run(
                dataset_id=dataset_id,
                target_label="local",
                started_at=STARTED_AT,
            )

        async with session_factory.begin() as session:
            return await RunRepository(session).abort_run(
                created.run_id,
                completed_at=ABORTED_AT,
            )
    finally:
        await engine.dispose()


def test_finished_runは再度completeできない(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    snapshot = asyncio.run(run_complete_completed_run(database_path))

    assert snapshot is None


async def run_complete_completed_run(database_path: Path) -> RunSnapshot | None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset_id = await create_dataset(session)
            created = await RunRepository(session).create_run(
                dataset_id=dataset_id,
                target_label="local",
                started_at=STARTED_AT,
            )
            await RunRepository(session).complete_run(created.run_id, completed_at=COMPLETED_AT)

        async with session_factory.begin() as session:
            return await RunRepository(session).complete_run(
                created.run_id,
                completed_at=ABORTED_AT,
            )
    finally:
        await engine.dispose()


def test_get_run_missingならnoneを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    snapshot = asyncio.run(run_get_missing_run(database_path))

    assert snapshot is None


async def run_get_missing_run(database_path: Path) -> RunSnapshot | None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            return await RunRepository(session).get_run(999)
    finally:
        await engine.dispose()


def test_run_statusは3値だけを公開する() -> None:
    assert {status.value for status in RunStatus} == {
        "running",
        "completed",
        "aborted",
    }
