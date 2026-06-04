"""評価用の初期スキーマを作成する。"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None

CURRENT_TIMESTAMP = sa.text("CURRENT_TIMESTAMP")
RUNNING_RUN_INDEX = "uq_runs_running_dataset_target"


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("dataset_id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("dataset_version", sa.String(length=255), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=CURRENT_TIMESTAMP,
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=CURRENT_TIMESTAMP,
            nullable=False,
        ),
        sa.UniqueConstraint("name", "dataset_version", name="uq_datasets_name_version"),
    )
    op.create_table(
        "evaluation_cases",
        sa.Column("case_id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("case_key", sa.String(length=255), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("tags_json", sa.JSON(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=CURRENT_TIMESTAMP,
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=CURRENT_TIMESTAMP,
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.dataset_id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("dataset_id", "case_key", name="uq_evaluation_cases_dataset_case_key"),
        sa.CheckConstraint(
            "severity in ('low', 'medium', 'high')", name="ck_evaluation_cases_severity"
        ),
    )
    op.create_table(
        "assertions",
        sa.Column("assertion_db_id", sa.Integer(), primary_key=True),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("assertion_type", sa.String(length=64), nullable=False),
        sa.Column("expected_json", sa.JSON(), nullable=True),
        sa.Column("required", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("on_fail", sa.String(length=32), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=CURRENT_TIMESTAMP,
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=CURRENT_TIMESTAMP,
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["case_id"], ["evaluation_cases.case_id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("case_id", "id", name="uq_assertions_case_logical_id"),
        sa.CheckConstraint(
            "on_fail in ('fail', 'warn', 'needs_review')",
            name="ck_assertions_on_fail",
        ),
        sa.CheckConstraint("severity in ('low', 'medium', 'high')", name="ck_assertions_severity"),
    )
    op.create_table(
        "runs",
        sa.Column("run_id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("target_label", sa.String(length=255), nullable=False),
        sa.Column("target_version", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=CURRENT_TIMESTAMP,
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "last_heartbeat_at",
            sa.DateTime(timezone=True),
            server_default=CURRENT_TIMESTAMP,
            nullable=False,
        ),
        sa.Column("gate_config_snapshot_json", sa.JSON(), nullable=True),
        sa.Column("gate_result_json", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=CURRENT_TIMESTAMP,
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=CURRENT_TIMESTAMP,
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.dataset_id"], ondelete="RESTRICT"),
        sa.CheckConstraint("status in ('running', 'completed', 'aborted')", name="ck_runs_status"),
    )
    op.create_table(
        "case_results",
        sa.Column("case_result_id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("case_key", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_type", sa.String(length=64), nullable=True),
        sa.Column("evaluator_results_json", sa.JSON(), nullable=True),
        sa.Column("reviewer_verdict", sa.String(length=64), nullable=True),
        sa.Column("reviewer_note", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("final_status", sa.String(length=32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=CURRENT_TIMESTAMP,
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=CURRENT_TIMESTAMP,
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["case_id"], ["evaluation_cases.case_id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("run_id", "case_key", name="uq_case_results_run_case_key"),
        sa.CheckConstraint(
            "status in ('pass', 'failed', 'skipped', 'needs_review')",
            name="ck_case_results_status",
        ),
        sa.CheckConstraint(
            "error_type is null or error_type in "
            "('provider_error', 'overloaded', 'unknown_error', 'response_mode_mismatch')",
            name="ck_case_results_error_type",
        ),
        sa.CheckConstraint(
            "final_status is null or final_status in ('pass', 'failed')",
            name="ck_case_results_final_status",
        ),
    )
    op.create_table(
        "assertion_results",
        sa.Column("assertion_result_id", sa.Integer(), primary_key=True),
        sa.Column("case_result_id", sa.Integer(), nullable=False),
        sa.Column("assertion_db_id", sa.Integer(), nullable=False),
        sa.Column("assertion_id", sa.String(length=255), nullable=False),
        sa.Column("assertion_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("matched_value_json", sa.JSON(), nullable=True),
        sa.Column("expected_json", sa.JSON(), nullable=True),
        sa.Column("required", sa.Boolean(), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("on_fail", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=CURRENT_TIMESTAMP,
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["case_result_id"], ["case_results.case_result_id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["assertion_db_id"], ["assertions.assertion_db_id"], ondelete="RESTRICT"
        ),
        sa.UniqueConstraint(
            "case_result_id", "assertion_db_id", name="uq_assertion_results_case_assertion"
        ),
        sa.CheckConstraint(
            "status in ('pass', 'failed', 'warning', 'skipped', 'not_applicable')",
            name="ck_assertion_results_status",
        ),
        sa.CheckConstraint(
            "severity in ('low', 'medium', 'high')", name="ck_assertion_results_severity"
        ),
        sa.CheckConstraint(
            "on_fail in ('fail', 'warn', 'needs_review')",
            name="ck_assertion_results_on_fail",
        ),
    )
    op.create_index(
        RUNNING_RUN_INDEX,
        "runs",
        ["dataset_id", "target_label"],
        unique=True,
        sqlite_where=sa.text("status = 'running'"),
        postgresql_where=sa.text("status = 'running'"),
    )


def downgrade() -> None:
    op.drop_index(RUNNING_RUN_INDEX, table_name="runs")
    op.drop_table("assertion_results")
    op.drop_table("case_results")
    op.drop_table("runs")
    op.drop_table("assertions")
    op.drop_table("evaluation_cases")
    op.drop_table("datasets")
