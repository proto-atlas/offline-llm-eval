from collections.abc import Sequence
from dataclasses import dataclass

from offline_llm_eval.evaluator.status import CaseResultStatus

type CaseStatusValue = CaseResultStatus | str


@dataclass(frozen=True, slots=True)
class RunMetrics:
    total_count: int
    passed_count: int
    failed_count: int
    needs_review_count: int
    skipped_count: int
    executed_count: int
    pass_rate: float
    fail_rate: float
    skipped_ratio: float


def calculate_run_metrics(case_statuses: Sequence[CaseStatusValue]) -> RunMetrics:
    statuses = tuple(CaseResultStatus(status) for status in case_statuses)
    passed_count = _count_status(statuses, CaseResultStatus.PASS)
    failed_count = _count_status(statuses, CaseResultStatus.FAILED)
    needs_review_count = _count_status(statuses, CaseResultStatus.NEEDS_REVIEW)
    skipped_count = _count_status(statuses, CaseResultStatus.SKIPPED)
    executed_count = passed_count + failed_count + needs_review_count
    total_count = len(statuses)

    return RunMetrics(
        total_count=total_count,
        passed_count=passed_count,
        failed_count=failed_count,
        needs_review_count=needs_review_count,
        skipped_count=skipped_count,
        executed_count=executed_count,
        pass_rate=_divide_or_zero(passed_count, executed_count),
        fail_rate=_divide_or_zero(failed_count, executed_count),
        skipped_ratio=_divide_or_zero(skipped_count, total_count),
    )


def resolve_effective_case_status(
    status: CaseStatusValue,
    final_status: str | None,
) -> CaseResultStatus:
    case_status = CaseResultStatus(status)
    if case_status is CaseResultStatus.NEEDS_REVIEW and final_status is not None:
        return CaseResultStatus(final_status)
    return case_status


def _count_status(
    statuses: Sequence[CaseResultStatus],
    target_status: CaseResultStatus,
) -> int:
    return sum(1 for status in statuses if status is target_status)


def _divide_or_zero(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
