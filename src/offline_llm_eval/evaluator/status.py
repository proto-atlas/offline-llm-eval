from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from offline_llm_eval.dataset.assertion_model import (
    AssertionOnFail,
    AssertionSeverity,
)
from offline_llm_eval.evaluator.assertions import AssertionEvaluationStatus
from offline_llm_eval.evaluator.results_schema import (
    UnifiedAssertionType,
    UnifiedEvaluatorResultSchema,
)

FR007_PRIORITY_NEEDS_REVIEW: Final = 1
FR007_PRIORITY_HIGH_REQUIRED_FAILURE: Final = 2
FR007_PRIORITY_REQUIRED_FAILURE: Final = 3
FR007_PRIORITY_SKIPPED: Final = 4
FR007_PRIORITY_WARNING_ONLY: Final = 5
FR007_PRIORITY_PASS: Final = 6


class CaseResultStatus(StrEnum):
    PASS = "pass"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True, slots=True)
class CaseStatusDecision:
    status: CaseResultStatus
    priority: int
    detail: str


def resolve_case_status(
    *,
    case_severity: AssertionSeverity,
    evaluator_results: Sequence[UnifiedEvaluatorResultSchema],
    case_error_type: str | None = None,
) -> CaseStatusDecision:
    results = tuple(evaluator_results)
    normal_results = _normal_results(results)
    secret_scan_results = _secret_scan_results(results)

    if any(_is_needs_review_failure(result) for result in results):
        return CaseStatusDecision(
            status=CaseResultStatus.NEEDS_REVIEW,
            priority=FR007_PRIORITY_NEEDS_REVIEW,
            detail="needs_review_failure",
        )

    if case_severity is AssertionSeverity.HIGH and any(
        _is_required_fail_failure(result) for result in results
    ):
        return CaseStatusDecision(
            status=CaseResultStatus.FAILED,
            priority=FR007_PRIORITY_HIGH_REQUIRED_FAILURE,
            detail="high_case_required_failure",
        )

    if any(_is_required_fail_failure(result) for result in results):
        return CaseStatusDecision(
            status=CaseResultStatus.FAILED,
            priority=FR007_PRIORITY_REQUIRED_FAILURE,
            detail="required_failure",
        )

    if _matches_skipped_priority(
        normal_results=normal_results,
        secret_scan_results=secret_scan_results,
        has_case_error=case_error_type is not None,
    ):
        return CaseStatusDecision(
            status=CaseResultStatus.SKIPPED,
            priority=FR007_PRIORITY_SKIPPED,
            detail="all_normal_assertions_skipped_or_case_error",
        )

    if _matches_warning_only_priority(
        normal_results=normal_results,
        secret_scan_results=secret_scan_results,
    ):
        return CaseStatusDecision(
            status=CaseResultStatus.PASS,
            priority=FR007_PRIORITY_WARNING_ONLY,
            detail="warning_only",
        )

    return CaseStatusDecision(
        status=CaseResultStatus.PASS,
        priority=FR007_PRIORITY_PASS,
        detail="pass",
    )


def _normal_results(
    results: Sequence[UnifiedEvaluatorResultSchema],
) -> tuple[UnifiedEvaluatorResultSchema, ...]:
    return tuple(
        result
        for result in results
        if result.assertion_type is not UnifiedAssertionType.SECRET_SCAN
    )


def _secret_scan_results(
    results: Sequence[UnifiedEvaluatorResultSchema],
) -> tuple[UnifiedEvaluatorResultSchema, ...]:
    return tuple(
        result for result in results if result.assertion_type is UnifiedAssertionType.SECRET_SCAN
    )


def _is_needs_review_failure(result: UnifiedEvaluatorResultSchema) -> bool:
    return (
        result.status is AssertionEvaluationStatus.FAILED
        and result.on_fail is AssertionOnFail.NEEDS_REVIEW
    )


def _is_required_fail_failure(result: UnifiedEvaluatorResultSchema) -> bool:
    return (
        result.status is AssertionEvaluationStatus.FAILED
        and result.required
        and result.on_fail is AssertionOnFail.FAIL
    )


def _matches_skipped_priority(
    *,
    normal_results: Sequence[UnifiedEvaluatorResultSchema],
    secret_scan_results: Sequence[UnifiedEvaluatorResultSchema],
    has_case_error: bool,
) -> bool:
    if not _secret_scan_allows_skipped(secret_scan_results):
        return False

    if normal_results and all(_is_skipped_result(result) for result in normal_results):
        return True

    return not normal_results and has_case_error


def _matches_warning_only_priority(
    *,
    normal_results: Sequence[UnifiedEvaluatorResultSchema],
    secret_scan_results: Sequence[UnifiedEvaluatorResultSchema],
) -> bool:
    if not normal_results or not _secret_scan_all_pass(secret_scan_results):
        return False

    return all(_is_warning_result(result) for result in normal_results)


def _is_skipped_result(result: UnifiedEvaluatorResultSchema) -> bool:
    return result.status in {
        AssertionEvaluationStatus.SKIPPED,
        AssertionEvaluationStatus.NOT_APPLICABLE,
    }


def _is_warning_result(result: UnifiedEvaluatorResultSchema) -> bool:
    if result.status is AssertionEvaluationStatus.WARNING:
        return True

    return result.status is AssertionEvaluationStatus.FAILED and (
        not result.required or result.on_fail is AssertionOnFail.WARN
    )


def _secret_scan_allows_skipped(
    results: Sequence[UnifiedEvaluatorResultSchema],
) -> bool:
    return all(
        result.status in {AssertionEvaluationStatus.PASS, AssertionEvaluationStatus.NOT_APPLICABLE}
        for result in results
    )


def _secret_scan_all_pass(results: Sequence[UnifiedEvaluatorResultSchema]) -> bool:
    return all(
        result.status in {AssertionEvaluationStatus.PASS, AssertionEvaluationStatus.NOT_APPLICABLE}
        for result in results
    )
