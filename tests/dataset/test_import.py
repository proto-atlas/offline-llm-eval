import asyncio
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession

from offline_llm_eval.dataset import importer
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.util.yaml_loader import load_yaml_file


class DatasetImportResultLike(Protocol):
    dataset_id: int
    name: str
    dataset_version: str
    case_count: int
    assertion_count: int


class DatasetImportModule(Protocol):
    DatasetImportError: type[ValueError]

    async def import_dataset_document(
        self,
        session: AsyncSession,
        document: Mapping[str, object],
    ) -> DatasetImportResultLike: ...

    async def import_dataset_file(
        self,
        session: AsyncSession,
        path: Path,
    ) -> DatasetImportResultLike: ...


dataset_import = cast(DatasetImportModule, importer)


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def migrate_database(database_path: Path) -> None:
    command.upgrade(make_config(database_path), "head")


async def import_document(
    database_path: Path,
    document: Mapping[str, object],
) -> DatasetImportResultLike:
    engine = create_async_db_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            return await dataset_import.import_dataset_document(session, document)
    finally:
        await engine.dispose()


async def import_file(database_path: Path, path: Path) -> DatasetImportResultLike:
    engine = create_async_db_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            return await dataset_import.import_dataset_file(session, path)
    finally:
        await engine.dispose()


def fetch_one(database_path: Path, sql: str) -> tuple[object, ...]:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(sql).fetchone()

    assert row is not None
    return cast(tuple[object, ...], row)


def fetch_scalar(database_path: Path, sql: str) -> int:
    row = fetch_one(database_path, sql)
    value = row[0]
    assert isinstance(value, int)
    return value


def test_import_dataset_document_inserts_dataset_case_and_assertion_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    document = load_yaml_file(Path("datasets/initial/rag_basics.yaml"))

    result = asyncio.run(import_document(database_path, document))

    assert result.dataset_id == 1
    assert result.name == "rag_basics"
    assert result.dataset_version == "1.0.0"
    assert result.case_count == 2
    assert result.assertion_count == 4
    assert fetch_scalar(database_path, "select count(*) from datasets") == 1
    assert fetch_scalar(database_path, "select count(*) from evaluation_cases") == 2
    assert fetch_scalar(database_path, "select count(*) from assertions") == 4


def test_import_dataset_document_stores_case_fields(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    document = load_yaml_file(Path("datasets/initial/rag_basics.yaml"))

    asyncio.run(import_document(database_path, document))

    row = fetch_one(
        database_path,
        "select case_key, question, severity, tags_json, metadata_json, is_active "
        "from evaluation_cases where case_key = 'answer_with_single_source'",
    )
    assert row[0] == "answer_with_single_source"
    assert row[1] == "What does the run summary include?"
    assert row[2] == "medium"
    assert json.loads(cast(str, row[3])) == ["rag", "summary"]
    assert json.loads(cast(str, row[4])) == {"source_set": "api_docs"}
    assert row[5] == 1


def test_import_dataset_document_defaults_case_is_active_to_true(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    document = {
        "name": "minimal",
        "dataset_version": "1.0.0",
        "cases": [
            {
                "case_key": "case_one",
                "question": "Question?",
                "assertions": [],
            }
        ],
    }

    asyncio.run(import_document(database_path, document))

    assert fetch_scalar(database_path, "select is_active from evaluation_cases") == 1


def test_import_dataset_document_preserves_case_is_active_false(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    document = {
        "name": "inactive_case",
        "dataset_version": "1.0.0",
        "cases": [
            {
                "case_key": "case_one",
                "question": "Question?",
                "is_active": False,
                "assertions": [],
            }
        ],
    }

    asyncio.run(import_document(database_path, document))

    assert fetch_scalar(database_path, "select is_active from evaluation_cases") == 0


def test_import_dataset_file_accepts_json(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    json_path = tmp_path / "dataset.json"
    migrate_database(database_path)
    json_path.write_text(
        json.dumps(
            {
                "name": "json_dataset",
                "dataset_version": "1.0.0",
                "cases": [
                    {
                        "case_key": "case_one",
                        "question": "Question?",
                        "assertions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = asyncio.run(import_file(database_path, json_path))

    assert result.name == "json_dataset"
    assert result.case_count == 1


def test_import_initial_dataset_files_imports_all_cases(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    for path in sorted(Path("datasets/initial").glob("*.yaml")):
        asyncio.run(import_document(database_path, load_yaml_file(path)))

    assert fetch_scalar(database_path, "select count(*) from datasets") == 5
    assert fetch_scalar(database_path, "select count(*) from evaluation_cases") == 10
    assert fetch_scalar(database_path, "select count(*) from assertions") == 17


def test_import_dataset_document_rejects_duplicate_case_key(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    document = {
        "name": "duplicate_case",
        "dataset_version": "1.0.0",
        "cases": [
            {
                "case_key": "case_one",
                "question": "Question?",
                "assertions": [],
            },
            {
                "case_key": "case_one",
                "question": "Another question?",
                "assertions": [],
            },
        ],
    }

    with pytest.raises(ValueError, match="case_key はデータセット内で一意"):
        asyncio.run(import_document(database_path, document))
