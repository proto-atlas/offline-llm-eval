from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from offline_llm_eval.cli.gate_config_schema import GateConfigSchema
from offline_llm_eval.dataset.assertion_model import AssertionOnFail, AssertionSeverity
from offline_llm_eval.diff.comparator import RunMetricsDelta
from offline_llm_eval.evaluator.assertions import AssertionEvaluationStatus
from offline_llm_eval.evaluator.secret_pattern import SECRET_SCAN_ASSERTION_ID
from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.run.metrics import RunMetrics


class HighSeverityFailureReason(StrEnum):
    HIGH_SEVERITY_ASSERTION_FAILED = "high_severity_assertion_failed"
    SECRET_SCAN_FAILED = "secret_scan_failed"


class ThresholdCriterion(StrEnum):
    PASS_RATE_MIN = "pass_rate_min"
    FAIL_RATE_MAX = "fail_rate_max"
    MAX_PASS_RATE_DELTA = "max_pass_rate_delta"
    MAX_FAIL_RATE_DELTA = "max_fail_rate_delta"
    MAX_SKIPPED_RATIO_DELTA = "max_skipped_ratio_delta"
    MAX_HIGH_SEVERITY_SKIPPED = "max_high_severity_skipped"


class ThresholdStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GateRulesVerdict(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


type PassesThreshold = Callable[[float | int, float | int], bool]


@dataclass(frozen=True, slots=True)
class GateAssertionResult:
    assertion_id: str
    status: AssertionEvaluationStatus
    required: bool
    severity: AssertionSeverity
    on_fail: AssertionOnFail


@dataclass(frozen=True, slots=True)
class GateCaseResult:
    case_key: str
    assertions: tuple[GateAssertionResult, ...]
    status: CaseResultStatus | None = None
    case_severity: AssertionSeverity = AssertionSeverity.MEDIUM
    final_status: str | None = None


@dataclass(frozen=True, slots=True)
class HighSeverityFailure:
    case_key: str
    assertion_id: str
    reason: HighSeverityFailureReason


@dataclass(frozen=True, slots=True)
class HighSeverityMustPassEvaluation:
    passed: bool
    failures: tuple[HighSeverityFailure, ...]


@dataclass(frozen=True, slots=True)
class SecretLeakEvaluation:
    passed: bool
    failures: tuple[HighSeverityFailure, ...]


@dataclass(frozen=True, slots=True)
class ThresholdEvaluation:
    criterion: ThresholdCriterion
    status: ThresholdStatus
    actual: float | int | None
    threshold: float | int
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class GateRulesEvaluation:
    verdict: GateRulesVerdict
    high_severity_must_pass: HighSeverityMustPassEvaluation | None
    fail_on_secret_leak: SecretLeakEvaluation | None
    thresholds: tuple[ThresholdEvaluation, ...]


def evaluate_high_severity_must_pass(
    cases: Sequence[GateCaseResult],
) -> HighSeverityMustPassEvaluation:
    failures = tuple(failure for case in cases for failure in _collect_high_severity_failures(case))
    return HighSeverityMustPassEvaluation(
        passed=not failures,
        failures=failures,
    )


def evaluate_fail_on_secret_leak(
    cases: Sequence[GateCaseResult],
) -> SecretLeakEvaluation:
    failures = tuple(failure for case in cases for failure in _collect_secret_scan_failures(case))
    return SecretLeakEvaluation(
        passed=not failures,
        failures=failures,
    )


def evaluate_gate_rules(
    config: GateConfigSchema,
    *,
    current_metrics: RunMetrics,
    metrics_delta: RunMetricsDelta | None,
    cases: Sequence[GateCaseResult] = (),
) -> GateRulesEvaluation:
    high_severity_evaluation = _evaluate_optional_high_severity(config, cases)
    secret_leak_evaluation = _evaluate_optional_secret_leak(config, cases)
    threshold_evaluations = evaluate_thresholds(
        config,
        current_metrics=current_metrics,
        metrics_delta=metrics_delta,
        cases=cases,
    )
    return GateRulesEvaluation(
        verdict=_gate_rules_verdict(
            high_severity_evaluation=high_severity_evaluation,
            secret_leak_evaluation=secret_leak_evaluation,
            threshold_evaluations=threshold_evaluations,
        ),
        high_severity_must_pass=high_severity_evaluation,
        fail_on_secret_leak=secret_leak_evaluation,
        thresholds=threshold_evaluations,
    )


def evaluate_thresholds(
    config: GateConfigSchema,
    *,
    current_metrics: RunMetrics,
    metrics_delta: RunMetricsDelta | None,
    cases: Sequence[GateCaseResult] = (),
) -> tuple[ThresholdEvaluation, ...]:
    evaluations: list[ThresholdEvaluation] = []
    _append_optional_threshold(
        evaluations,
        criterion=ThresholdCriterion.PASS_RATE_MIN,
        actual=current_metrics.pass_rate,
        threshold=config.pass_rate_min,
        passes=lambda actual, threshold: actual >= threshold,
    )
    _append_optional_threshold(
        evaluations,
        criterion=ThresholdCriterion.FAIL_RATE_MAX,
        actual=current_metrics.fail_rate,
        threshold=config.fail_rate_max,
        passes=lambda actual, threshold: actual <= threshold,
    )
    _append_delta_thresholds(evaluations, config=config, metrics_delta=metrics_delta)
    _append_optional_threshold(
        evaluations,
        criterion=ThresholdCriterion.MAX_HIGH_SEVERITY_SKIPPED,
        actual=_count_high_severity_skipped_cases(cases),
        threshold=config.max_high_severity_skipped,
        passes=lambda actual, threshold: actual <= threshold,
    )
    return tuple(evaluations)


def _evaluate_optional_high_severity(
    config: GateConfigSchema,
    cases: Sequence[GateCaseResult],
) -> HighSeverityMustPassEvaluation | None:
    if not config.high_severity_must_pass:
        return None
    return evaluate_high_severity_must_pass(cases)


def _evaluate_optional_secret_leak(
    config: GateConfigSchema,
    cases: Sequence[GateCaseResult],
) -> SecretLeakEvaluation | None:
    if not config.fail_on_secret_leak:
        return None
    return evaluate_fail_on_secret_leak(cases)


def _gate_rules_verdict(
    *,
    high_severity_evaluation: HighSeverityMustPassEvaluation | None,
    secret_leak_evaluation: SecretLeakEvaluation | None,
    threshold_evaluations: Sequence[ThresholdEvaluation],
) -> GateRulesVerdict:
    if _has_rule_failure(
        high_severity_evaluation=high_severity_evaluation,
        secret_leak_evaluation=secret_leak_evaluation,
        threshold_evaluations=threshold_evaluations,
    ):
        return GateRulesVerdict.FAILED
    return GateRulesVerdict.PASSED


def _has_rule_failure(
    *,
    high_severity_evaluation: HighSeverityMustPassEvaluation | None,
    secret_leak_evaluation: SecretLeakEvaluation | None,
    threshold_evaluations: Sequence[ThresholdEvaluation],
) -> bool:
    return (
        _optional_evaluation_failed(high_severity_evaluation)
        or _optional_evaluation_failed(secret_leak_evaluation)
        or any(evaluation.status is ThresholdStatus.FAILED for evaluation in threshold_evaluations)
    )


def _optional_evaluation_failed(
    evaluation: HighSeverityMustPassEvaluation | SecretLeakEvaluation | None,
) -> bool:
    return evaluation is not None and not evaluation.passed


def _append_delta_thresholds(
    evaluations: list[ThresholdEvaluation],
    *,
    config: GateConfigSchema,
    metrics_delta: RunMetricsDelta | None,
) -> None:
    _append_optional_delta_threshold(
        evaluations,
        criterion=ThresholdCriterion.MAX_PASS_RATE_DELTA,
        actual=_pass_rate_drop(metrics_delta),
        threshold=config.max_pass_rate_delta,
    )
    _append_optional_delta_threshold(
        evaluations,
        criterion=ThresholdCriterion.MAX_FAIL_RATE_DELTA,
        actual=_fail_rate_increase(metrics_delta),
        threshold=config.max_fail_rate_delta,
    )
    _append_optional_delta_threshold(
        evaluations,
        criterion=ThresholdCriterion.MAX_SKIPPED_RATIO_DELTA,
        actual=_skipped_ratio_increase(metrics_delta),
        threshold=config.max_skipped_ratio_delta,
    )


def _append_optional_threshold(
    evaluations: list[ThresholdEvaluation],
    *,
    criterion: ThresholdCriterion,
    actual: float | int,
    threshold: float | int | None,
    passes: PassesThreshold,
) -> None:
    if threshold is None:
        return

    evaluations.append(
        ThresholdEvaluation(
            criterion=criterion,
            status=_threshold_status(passes(actual, threshold)),
            actual=actual,
            threshold=threshold,
        )
    )


def _append_optional_delta_threshold(
    evaluations: list[ThresholdEvaluation],
    *,
    criterion: ThresholdCriterion,
    actual: float | None,
    threshold: float | None,
) -> None:
    if threshold is None:
        return

    if actual is None:
        evaluations.append(
            ThresholdEvaluation(
                criterion=criterion,
                status=ThresholdStatus.SKIPPED,
                actual=None,
                threshold=threshold,
                reason="baseline_not_available",
            )
        )
        return

    evaluations.append(
        ThresholdEvaluation(
            criterion=criterion,
            status=_threshold_status(actual <= threshold),
            actual=actual,
            threshold=threshold,
        )
    )


def _threshold_status(passed: bool) -> ThresholdStatus:
    if passed:
        return ThresholdStatus.PASSED
    return ThresholdStatus.FAILED


def _pass_rate_drop(metrics_delta: RunMetricsDelta | None) -> float | None:
    if metrics_delta is None:
        return None
    return max(0.0, -metrics_delta.pass_rate)


def _fail_rate_increase(metrics_delta: RunMetricsDelta | None) -> float | None:
    if metrics_delta is None:
        return None
    return max(0.0, metrics_delta.fail_rate)


def _skipped_ratio_increase(metrics_delta: RunMetricsDelta | None) -> float | None:
    if metrics_delta is None:
        return None
    return max(0.0, metrics_delta.skipped_ratio)


def _count_high_severity_skipped_cases(cases: Sequence[GateCaseResult]) -> int:
    return sum(
        1
        for case in cases
        if case.case_severity is AssertionSeverity.HIGH and case.status is CaseResultStatus.SKIPPED
    )


def _collect_high_severity_failures(
    case: GateCaseResult,
) -> tuple[HighSeverityFailure, ...]:
    return tuple(
        HighSeverityFailure(
            case_key=case.case_key,
            assertion_id=assertion.assertion_id,
            reason=_high_severity_failure_reason(assertion),
        )
        for assertion in case.assertions
        if _is_high_severity_failure(assertion)
    )


def _collect_secret_scan_failures(
    case: GateCaseResult,
) -> tuple[HighSeverityFailure, ...]:
    return tuple(
        HighSeverityFailure(
            case_key=case.case_key,
            assertion_id=assertion.assertion_id,
            reason=HighSeverityFailureReason.SECRET_SCAN_FAILED,
        )
        for assertion in case.assertions
        if _is_secret_scan_failure(assertion)
    )


def _is_high_severity_failure(assertion: GateAssertionResult) -> bool:
    if assertion.assertion_id == SECRET_SCAN_ASSERTION_ID:
        return _is_secret_scan_failure(assertion)

    return (
        assertion.status is AssertionEvaluationStatus.FAILED
        and assertion.required
        and assertion.severity is AssertionSeverity.HIGH
        and assertion.on_fail is AssertionOnFail.FAIL
    )


def _is_secret_scan_failure(assertion: GateAssertionResult) -> bool:
    return (
        assertion.assertion_id == SECRET_SCAN_ASSERTION_ID
        and assertion.status is AssertionEvaluationStatus.FAILED
    )


def _high_severity_failure_reason(
    assertion: GateAssertionResult,
) -> HighSeverityFailureReason:
    if assertion.assertion_id == SECRET_SCAN_ASSERTION_ID:
        return HighSeverityFailureReason.SECRET_SCAN_FAILED
    return HighSeverityFailureReason.HIGH_SEVERITY_ASSERTION_FAILED
