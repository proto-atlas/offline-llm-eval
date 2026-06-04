import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

EXPECTED_TABLES = {
    "alembic_version",
    "assertion_results",
    "assertions",
    "case_results",
    "datasets",
    "evaluation_cases",
    "runs",
}


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def test_alembic_upgrade_head_creates_initial_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"

    command.upgrade(make_config(database_path), "head")

    with sqlite3.connect(database_path) as connection:
        table_rows = connection.execute(
            "select name from sqlite_master where type = 'table'"
        ).fetchall()

    assert {row[0] for row in table_rows} == EXPECTED_TABLES


def test_alembic_upgrade_head_creates_running_run_partial_unique_index(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"

    command.upgrade(make_config(database_path), "head")

    with sqlite3.connect(database_path) as connection:
        index_sql = connection.execute(
            "select sql from sqlite_master where type = 'index' and name = ?",
            ("uq_runs_running_dataset_target",),
        ).fetchone()

    assert index_sql is not None
    assert index_sql[0] == (
        "CREATE UNIQUE INDEX uq_runs_running_dataset_target "
        "ON runs (dataset_id, target_label) "
        "WHERE status = 'running'"
    )
