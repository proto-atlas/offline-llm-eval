from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    text,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column

from offline_llm_eval.db.base import Base

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


class AssertionType(StrEnum):
    EXACT_MATCH = "exact_match"
    NORMALIZED_CONTAINS = "normalized_contains"
    KEYWORD_ALL = "keyword_all"
    KEYWORD_ANY = "keyword_any"
    REGEX = "regex"
    NO_ANSWER_EXPECTED = "no_answer_expected"
    CITATION_PRESENCE = "citation_presence"
    SOURCE_ID_EXACT_SET = "source_id_exact_set"
    SOURCE_ID_SUBSET = "source_id_subset"
    JSON_SCHEMA = "json_schema"
    FORBIDDEN_PHRASE = "forbidden_phrase"
    LATENCY_THRESHOLD = "latency_threshold"


class AssertionOnFail(StrEnum):
    FAIL = "fail"
    WARN = "warn"
    NEEDS_REVIEW = "needs_review"


class AssertionSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AssertionYaml(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    assertion_type: AssertionType = Field(alias="type")
    expected: JsonValue = None
    required: bool = True
    on_fail: AssertionOnFail = AssertionOnFail.FAIL
    severity: AssertionSeverity | None = None
    is_active: bool = True

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if value == "":
            raise ValueError("assertion id は空にできません。")

        if value.strip() != value:
            raise ValueError("assertion id は前後に空白を含められません。")

        return value


class AssertionRecord(Base):
    __tablename__ = "assertions"
    __table_args__ = (
        UniqueConstraint("case_id", "id", name="uq_assertions_case_logical_id"),
        CheckConstraint(
            "on_fail in ('fail', 'warn', 'needs_review')",
            name="ck_assertions_on_fail",
        ),
        CheckConstraint(
            "severity in ('low', 'medium', 'high')",
            name="ck_assertions_severity",
        ),
    )

    assertion_db_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(
        ForeignKey("evaluation_cases.case_id", ondelete="RESTRICT"),
        nullable=False,
    )
    id: Mapped[str] = mapped_column(String(255), nullable=False)
    assertion_type: Mapped[str] = mapped_column(String(64), nullable=False)
    expected_json: Mapped[JsonValue] = mapped_column(JSON, nullable=True)
    required: Mapped[bool] = mapped_column(Boolean, server_default=true(), nullable=False)
    on_fail: Mapped[str] = mapped_column(String(32), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=true(), nullable=False)
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


def resolve_assertion_severity(
    assertion: AssertionYaml,
    case_severity: AssertionSeverity,
) -> AssertionSeverity:
    return assertion.severity or case_severity
