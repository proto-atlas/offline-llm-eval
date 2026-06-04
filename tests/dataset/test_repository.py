import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from offline_llm_eval.dataset.repository import Dataset, DatasetRepository
from offline_llm_eval.db import create_async_db_engine, create_session_factory


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def migrate_database(database_path: Path) -> None:
    command.upgrade(make_config(database_path), "head")


async def count_datasets(session_factory: async_sessionmaker[AsyncSession]) -> int:
    async with session_factory() as session:
        result = await session.execute(select(func.count()).select_from(Dataset))

    return int(result.scalar_one())


def create_test_engine(database_path: Path) -> AsyncEngine:
    return create_async_db_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")


def test_create_dataset_assigns_dataset_id(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    asyncio.run(run_create_dataset_assigns_dataset_id(database_path))


async def run_create_dataset_assigns_dataset_id(database_path: Path) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            repository = DatasetRepository(session)

            record = await repository.create_dataset(
                "rag_basics",
                "1.0.0",
                metadata_json={"source": "initial"},
            )

        assert record.dataset_id == 1
        assert record.name == "rag_basics"
        assert record.dataset_version == "1.0.0"
        assert record.metadata_json == {"source": "initial"}
    finally:
        await engine.dispose()


def test_get_or_create_dataset_reuses_same_name_and_version(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    asyncio.run(run_get_or_create_dataset_reuses_same_name_and_version(database_path))


async def run_get_or_create_dataset_reuses_same_name_and_version(database_path: Path) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            repository = DatasetRepository(session)

            first = await repository.get_or_create_dataset("rag_basics", "1.0.0")
            second = await repository.get_or_create_dataset("rag_basics", "1.0.0")

        assert first.dataset_id == 1
        assert second.dataset_id == 1
        assert await count_datasets(session_factory) == 1
    finally:
        await engine.dispose()


def test_get_or_create_dataset_creates_new_id_for_different_version(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    asyncio.run(run_get_or_create_dataset_creates_new_id_for_different_version(database_path))


async def run_get_or_create_dataset_creates_new_id_for_different_version(
    database_path: Path,
) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            repository = DatasetRepository(session)

            first = await repository.get_or_create_dataset("rag_basics", "1.0.0")
            second = await repository.get_or_create_dataset("rag_basics", "1.1.0")

        assert first.dataset_id == 1
        assert second.dataset_id == 2
        assert await count_datasets(session_factory) == 2
    finally:
        await engine.dispose()


def test_create_dataset_rejects_duplicate_name_and_version(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    asyncio.run(run_create_dataset_rejects_duplicate_name_and_version(database_path))


async def run_create_dataset_rejects_duplicate_name_and_version(database_path: Path) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            repository = DatasetRepository(session)

            await repository.create_dataset("rag_basics", "1.0.0")
            with pytest.raises(IntegrityError):
                await repository.create_dataset("rag_basics", "1.0.0")
    finally:
        await engine.dispose()
