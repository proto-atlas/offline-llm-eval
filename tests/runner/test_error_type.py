import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from offline_llm_eval.provider.mock_behavior import (
    MockErrorCode,
    MockErrorResponse,
    parse_mock_response,
)
from offline_llm_eval.runner.error_type import (
    CASE_RESULT_ERROR_TYPE_CHECK_NAME,
    CASE_RESULT_ERROR_TYPE_CHECK_SQL,
    ERROR_TYPE_VALUES,
    ErrorType,
    InvalidErrorTypeError,
    parse_error_type,
    parse_optional_error_type,
)


def test_error_type_values_match_fr_034() -> None:
    assert ERROR_TYPE_VALUES == (
        "provider_error",
        "overloaded",
        "unknown_error",
        "response_mode_mismatch",
    )


@pytest.mark.parametrize("value", ERROR_TYPE_VALUES)
def test_parse_error_type_accepts_fr_034_values(value: str) -> None:
    assert parse_error_type(value) == ErrorType(value)


def test_parse_error_type_rejects_unknown_value() -> None:
    with pytest.raises(InvalidErrorTypeError) as error:
        parse_error_type("timeout")

    assert error.value.code == "validation_error"


def test_parse_optional_error_type_accepts_none() -> None:
    assert parse_optional_error_type(None) is None


def test_mock_error_code_is_error_type_alias() -> None:
    assert MockErrorCode is ErrorType


def test_mock_response_error_uses_error_type() -> None:
    behavior = parse_mock_response(
        {
            "error": {
                "code": "provider_error",
                "message": "providerが失敗を返しました。",
            }
        }
    )

    assert isinstance(behavior.response, MockErrorResponse)
    assert behavior.response.error.code is ErrorType.PROVIDER_ERROR


def test_case_result_error_type_check_sql_matches_migration() -> None:
    assert CASE_RESULT_ERROR_TYPE_CHECK_SQL == (
        "error_type is null or error_type in "
        "('provider_error', 'overloaded', 'unknown_error', 'response_mode_mismatch')"
    )


def test_case_result_error_type_check_exists_after_migration(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"

    command.upgrade(_make_config(database_path), "head")

    with sqlite3.connect(database_path) as connection:
        create_table_sql = connection.execute(
            "select sql from sqlite_master where type = 'table' and name = ?",
            ("case_results",),
        ).fetchone()

    assert create_table_sql is not None
    assert CASE_RESULT_ERROR_TYPE_CHECK_NAME in create_table_sql[0]
    assert CASE_RESULT_ERROR_TYPE_CHECK_SQL in create_table_sql[0]


def test_case_result_error_type_check_accepts_fr_034_values(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"

    command.upgrade(_make_config(database_path), "head")

    with sqlite3.connect(database_path) as connection:
        _insert_case_result_dependencies(connection)
        for index, error_type in enumerate(ERROR_TYPE_VALUES, start=1):
            connection.execute(
                """
                insert into case_results
                    (run_id, case_id, case_key, status, error_type)
                values
                    (1, 1, ?, 'skipped', ?)
                """,
                (f"case_{index}", error_type),
            )

        rows = connection.execute("select count(*) from case_results").fetchone()

    assert rows == (4,)


def test_case_result_error_type_check_accepts_null(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"

    command.upgrade(_make_config(database_path), "head")

    with sqlite3.connect(database_path) as connection:
        _insert_case_result_dependencies(connection)
        connection.execute(
            """
            insert into case_results
                (run_id, case_id, case_key, status, error_type)
            values
                (1, 1, 'case_without_error', 'pass', null)
            """,
        )

        row = connection.execute("select error_type from case_results").fetchone()

    assert row == (None,)


def test_case_result_error_type_check_rejects_unknown_value(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"

    command.upgrade(_make_config(database_path), "head")

    with sqlite3.connect(database_path) as connection:
        _insert_case_result_dependencies(connection)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                insert into case_results
                    (run_id, case_id, case_key, status, error_type)
                values
                    (1, 1, 'case_timeout', 'skipped', 'timeout')
                """,
            )


def _make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def _insert_case_result_dependencies(connection: sqlite3.Connection) -> None:
    connection.execute(
        "insert into datasets (dataset_id, name, dataset_version) values (1, 'demo', '1.0.0')"
    )
    connection.execute(
        """
        insert into evaluation_cases
            (case_id, dataset_id, case_key, question, severity)
        values
            (1, 1, 'case_1', 'Question?', 'medium')
        """
    )
    connection.execute(
        """
        insert into runs
            (run_id, dataset_id, target_label, status)
        values
            (1, 1, 'local', 'completed')
        """
    )
