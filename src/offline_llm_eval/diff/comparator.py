from collections.abc import Sequence
from dataclasses import dataclass

from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.run.metrics import RunMetrics, calculate_run_metrics

type CaseStatusValue = CaseResultStatus | str


@dataclass(frozen=True, slots=True)
class CaseResultForComparison:
    case_key: str
    status: CaseStatusValue


@dataclass(frozen=True, slots=True)
class CaseKeyComparison:
    shared_case_keys: tuple[str, ...]
    added_case_keys: tuple[str, ...]
    removed_case_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunMetricsDelta:
    total_count: int
    passed_count: int
    failed_count: int
    needs_review_count: int
    skipped_count: int
    executed_count: int
    pass_rate: float
    fail_rate: float
    skipped_ratio: float


@dataclass(frozen=True, slots=True)
class RunMetricsComparison:
    baseline: RunMetrics
    current: RunMetrics
    delta: RunMetricsDelta


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    case_keys: CaseKeyComparison
    metrics: RunMetricsComparison


def compare_runs(
    *,
    baseline_cases: Sequence[CaseResultForComparison],
    current_cases: Sequence[CaseResultForComparison],
) -> BaselineComparison:
    baseline_metrics = calculate_run_metrics(_case_statuses(baseline_cases))
    current_metrics = calculate_run_metrics(_case_statuses(current_cases))
    return BaselineComparison(
        case_keys=compare_case_keys(
            baseline_case_keys=_case_keys(baseline_cases),
            current_case_keys=_case_keys(current_cases),
        ),
        metrics=RunMetricsComparison(
            baseline=baseline_metrics,
            current=current_metrics,
            delta=_calculate_metrics_delta(
                baseline=baseline_metrics,
                current=current_metrics,
            ),
        ),
    )


def compare_case_keys(
    *,
    baseline_case_keys: Sequence[str],
    current_case_keys: Sequence[str],
) -> CaseKeyComparison:
    baseline_keys = set(baseline_case_keys)
    current_keys = set(current_case_keys)
    return CaseKeyComparison(
        shared_case_keys=tuple(sorted(baseline_keys & current_keys)),
        added_case_keys=tuple(sorted(current_keys - baseline_keys)),
        removed_case_keys=tuple(sorted(baseline_keys - current_keys)),
    )


def _case_keys(cases: Sequence[CaseResultForComparison]) -> tuple[str, ...]:
    return tuple(case.case_key for case in cases)


def _case_statuses(
    cases: Sequence[CaseResultForComparison],
) -> tuple[CaseStatusValue, ...]:
    return tuple(case.status for case in cases)


def _calculate_metrics_delta(
    *,
    baseline: RunMetrics,
    current: RunMetrics,
) -> RunMetricsDelta:
    return RunMetricsDelta(
        total_count=current.total_count - baseline.total_count,
        passed_count=current.passed_count - baseline.passed_count,
        failed_count=current.failed_count - baseline.failed_count,
        needs_review_count=current.needs_review_count - baseline.needs_review_count,
        skipped_count=current.skipped_count - baseline.skipped_count,
        executed_count=current.executed_count - baseline.executed_count,
        pass_rate=current.pass_rate - baseline.pass_rate,
        fail_rate=current.fail_rate - baseline.fail_rate,
        skipped_ratio=current.skipped_ratio - baseline.skipped_ratio,
    )
