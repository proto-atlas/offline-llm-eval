from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from offline_llm_eval.run.repository import RunSnapshot


class DiffWarningCode(StrEnum):
    CASE_UPDATED_AFTER_BASELINE = "case_updated_after_baseline"


@dataclass(frozen=True, slots=True)
class CaseUpdatedAt:
    case_key: str
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class DiffWarning:
    code: DiffWarningCode
    case_key: str
    baseline_run_id: int
    baseline_started_at: datetime
    case_updated_at: datetime


def collect_case_updated_at_warnings(
    *,
    baseline_run: RunSnapshot,
    cases: Sequence[CaseUpdatedAt],
) -> tuple[DiffWarning, ...]:
    return tuple(
        DiffWarning(
            code=DiffWarningCode.CASE_UPDATED_AFTER_BASELINE,
            case_key=case.case_key,
            baseline_run_id=baseline_run.run_id,
            baseline_started_at=baseline_run.started_at,
            case_updated_at=case.updated_at,
        )
        for case in sorted(cases, key=lambda item: item.case_key)
        if case.updated_at > baseline_run.started_at
    )
