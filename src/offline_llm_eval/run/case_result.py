from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from offline_llm_eval.dataset.repository import JsonValue
from offline_llm_eval.db.base import Base
from offline_llm_eval.runner.error_type import CASE_RESULT_ERROR_TYPE_CHECK_SQL


class CaseResultRecord(Base):
    __tablename__ = "case_results"
    __table_args__ = (
        UniqueConstraint("run_id", "case_key", name="uq_case_results_run_case_key"),
        CheckConstraint(
            "status in ('pass', 'failed', 'skipped', 'needs_review')",
            name="ck_case_results_status",
        ),
        CheckConstraint(
            CASE_RESULT_ERROR_TYPE_CHECK_SQL,
            name="ck_case_results_error_type",
        ),
        CheckConstraint(
            "final_status is null or final_status in ('pass', 'failed')",
            name="ck_case_results_final_status",
        ),
    )

    case_result_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False)
    case_id: Mapped[int] = mapped_column(Integer, nullable=False)
    case_key: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evaluator_results_json: Mapped[JsonValue | None] = mapped_column(JSON, nullable=True)
    reviewer_verdict: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    final_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


class AssertionResultRecord(Base):
    __tablename__ = "assertion_results"
    __table_args__ = (
        UniqueConstraint(
            "case_result_id",
            "assertion_db_id",
            name="uq_assertion_results_case_assertion",
        ),
        CheckConstraint(
            "status in ('pass', 'failed', 'warning', 'skipped', 'not_applicable')",
            name="ck_assertion_results_status",
        ),
        CheckConstraint(
            "severity in ('low', 'medium', 'high')",
            name="ck_assertion_results_severity",
        ),
        CheckConstraint(
            "on_fail in ('fail', 'warn', 'needs_review')",
            name="ck_assertion_results_on_fail",
        ),
    )

    assertion_result_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_result_id: Mapped[int] = mapped_column(Integer, nullable=False)
    assertion_db_id: Mapped[int] = mapped_column(Integer, nullable=False)
    assertion_id: Mapped[str] = mapped_column(String(255), nullable=False)
    assertion_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    matched_value_json: Mapped[JsonValue | None] = mapped_column(JSON, nullable=True)
    expected_json: Mapped[JsonValue | None] = mapped_column(JSON, nullable=True)
    required: Mapped[bool] = mapped_column(nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    on_fail: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
