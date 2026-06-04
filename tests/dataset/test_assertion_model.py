import sqlite3
from pathlib import Path
from typing import cast

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import CheckConstraint, Table, UniqueConstraint

from offline_llm_eval.dataset.assertion_model import (
    AssertionOnFail,
    AssertionRecord,
    AssertionSeverity,
    AssertionType,
    AssertionYaml,
    resolve_assertion_severity,
)


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def test_assertion_yaml_accepts_required_fields_and_defaults() -> None:
    assertion = AssertionYaml.model_validate(
        {
            "id": "answer_exact",
            "type": "exact_match",
            "expected": "expected answer",
        }
    )

    assert assertion.id == "answer_exact"
    assert assertion.assertion_type is AssertionType.EXACT_MATCH
    assert assertion.expected == "expected answer"
    assert assertion.required is True
    assert assertion.on_fail is AssertionOnFail.FAIL
    assert assertion.severity is None
    assert assertion.is_active is True
    assert resolve_assertion_severity(assertion, AssertionSeverity.HIGH) is AssertionSeverity.HIGH


def test_assertion_yaml_accepts_optional_controls() -> None:
    assertion = AssertionYaml.model_validate(
        {
            "id": "latency_under_threshold",
            "type": "latency_threshold",
            "expected": 1500,
            "required": False,
            "on_fail": "warn",
            "severity": "low",
            "is_active": False,
        }
    )

    assert assertion.assertion_type is AssertionType.LATENCY_THRESHOLD
    assert assertion.required is False
    assert assertion.on_fail is AssertionOnFail.WARN
    assert assertion.severity is AssertionSeverity.LOW
    assert assertion.is_active is False
    assert resolve_assertion_severity(assertion, AssertionSeverity.HIGH) is AssertionSeverity.LOW


def test_assertion_yaml_allows_python_field_name_for_internal_callers() -> None:
    assertion = AssertionYaml.model_validate(
        {
            "id": "contains_required_terms",
            "assertion_type": "keyword_all",
            "expected": ["run metadata", "counts"],
        }
    )

    assert assertion.assertion_type is AssertionType.KEYWORD_ALL


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "", "type": "exact_match"},
        {"id": " answer_exact", "type": "exact_match"},
    ],
)
def test_assertion_yaml_rejects_invalid_logical_id(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="assertion id"):
        AssertionYaml.model_validate(payload)


def test_assertion_yaml_rejects_unknown_assertion_type() -> None:
    with pytest.raises(ValidationError, match="Input should be"):
        AssertionYaml.model_validate({"id": "unknown", "type": "unknown_type"})


def test_assertion_yaml_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AssertionYaml.model_validate(
            {
                "id": "answer_exact",
                "type": "exact_match",
                "unknown": True,
            }
        )


def test_assertion_record_table_matches_db_contract() -> None:
    table = cast(Table, AssertionRecord.__table__)

    assert table.name == "assertions"
    assert [column.name for column in table.primary_key.columns] == ["assertion_db_id"]
    assert table.c.id.primary_key is False
    assert set(table.c.keys()) == {
        "assertion_db_id",
        "case_id",
        "id",
        "assertion_type",
        "expected_json",
        "required",
        "on_fail",
        "severity",
        "is_active",
        "created_at",
        "updated_at",
    }

    unique_constraints = {
        constraint.name: tuple(column.name for column in constraint.columns)
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    check_constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert unique_constraints == {"uq_assertions_case_logical_id": ("case_id", "id")}
    assert check_constraints == {
        "ck_assertions_on_fail": "on_fail in ('fail', 'warn', 'needs_review')",
        "ck_assertions_severity": "severity in ('low', 'medium', 'high')",
    }


def test_migration_uses_assertion_db_id_for_assertion_result_fk(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"

    command.upgrade(make_config(database_path), "head")

    with sqlite3.connect(database_path) as connection:
        assertion_fks = connection.execute("pragma foreign_key_list(assertion_results)").fetchall()
        assertion_table_sql = connection.execute(
            "select sql from sqlite_master where type = 'table' and name = ?",
            ("assertions",),
        ).fetchone()

    assert ("assertion_db_id", "assertions", "assertion_db_id") in {
        (row[3], row[2], row[4]) for row in assertion_fks
    }
    assert assertion_table_sql is not None
    assert "CONSTRAINT uq_assertions_case_logical_id UNIQUE (case_id, id)" in assertion_table_sql[0]
