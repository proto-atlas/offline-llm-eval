from offline_llm_eval.cli.gate_config_schema import GateConfigSchema
from offline_llm_eval.cli.gate_rules import (
    GateAssertionResult,
    GateCaseResult,
    GateRulesVerdict,
    HighSeverityFailure,
    HighSeverityFailureReason,
    ThresholdCriterion,
    ThresholdEvaluation,
    ThresholdStatus,
    evaluate_fail_on_secret_leak,
    evaluate_gate_rules,
    evaluate_high_severity_must_pass,
    evaluate_thresholds,
)
from offline_llm_eval.dataset.assertion_model import AssertionOnFail, AssertionSeverity
from offline_llm_eval.diff.comparator import RunMetricsDelta
from offline_llm_eval.evaluator.assertions import AssertionEvaluationStatus
from offline_llm_eval.evaluator.secret_pattern import SECRET_SCAN_ASSERTION_ID
from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.run.metrics import RunMetrics


def test_high_severity_must_pass() -> None:
    result = evaluate_high_severity_must_pass(
        [
            GateCaseResult(
                case_key="important_case",
                assertions=(
                    gate_assertion_result(
                        assertion_id="must_match",
                        status=AssertionEvaluationStatus.FAILED,
                        severity=AssertionSeverity.HIGH,
                        required=True,
                        on_fail=AssertionOnFail.FAIL,
                    ),
                ),
            )
        ]
    )

    assert result.failures == (
        HighSeverityFailure(
            case_key="important_case",
            assertion_id="must_match",
            reason=HighSeverityFailureReason.HIGH_SEVERITY_ASSERTION_FAILED,
        ),
    )


def test_high_severity_must_passはsecret_scan_failedを検知する() -> None:
    result = evaluate_high_severity_must_pass(
        [
            GateCaseResult(
                case_key="reviewed_case",
                final_status="pass",
                assertions=(
                    gate_assertion_result(
                        assertion_id=SECRET_SCAN_ASSERTION_ID,
                        status=AssertionEvaluationStatus.FAILED,
                        severity=AssertionSeverity.HIGH,
                    ),
                ),
            )
        ]
    )

    assert result.failures == (
        HighSeverityFailure(
            case_key="reviewed_case",
            assertion_id=SECRET_SCAN_ASSERTION_ID,
            reason=HighSeverityFailureReason.SECRET_SCAN_FAILED,
        ),
    )


def test_fail_on_secret_leakはsecret_scan_failedだけを検知する() -> None:
    result = evaluate_fail_on_secret_leak(
        [
            GateCaseResult(
                case_key="mixed_case",
                assertions=(
                    gate_assertion_result(
                        assertion_id=SECRET_SCAN_ASSERTION_ID,
                        status=AssertionEvaluationStatus.FAILED,
                        severity=AssertionSeverity.HIGH,
                    ),
                    gate_assertion_result(
                        assertion_id="high_failure",
                        status=AssertionEvaluationStatus.FAILED,
                        severity=AssertionSeverity.HIGH,
                    ),
                ),
            )
        ]
    )

    assert result.failures == (
        HighSeverityFailure(
            case_key="mixed_case",
            assertion_id=SECRET_SCAN_ASSERTION_ID,
            reason=HighSeverityFailureReason.SECRET_SCAN_FAILED,
        ),
    )


def test_high_severity_must_passはskippedを対象外にする() -> None:
    result = evaluate_high_severity_must_pass(
        [
            GateCaseResult(
                case_key="skipped_case",
                assertions=(
                    gate_assertion_result(
                        status=AssertionEvaluationStatus.SKIPPED,
                        severity=AssertionSeverity.HIGH,
                        required=True,
                        on_fail=AssertionOnFail.FAIL,
                    ),
                ),
            )
        ]
    )

    assert result.passed is True


def test_high_severity_must_passはwarningを対象外にする() -> None:
    result = evaluate_high_severity_must_pass(
        [
            GateCaseResult(
                case_key="warning_case",
                assertions=(
                    gate_assertion_result(
                        status=AssertionEvaluationStatus.WARNING,
                        severity=AssertionSeverity.HIGH,
                        required=False,
                        on_fail=AssertionOnFail.WARN,
                    ),
                ),
            )
        ]
    )

    assert result.passed is True


def test_high_severity_must_passは低severity失敗を対象外にする() -> None:
    result = evaluate_high_severity_must_pass(
        [
            GateCaseResult(
                case_key="low_case",
                assertions=(
                    gate_assertion_result(
                        status=AssertionEvaluationStatus.FAILED,
                        severity=AssertionSeverity.LOW,
                        required=True,
                        on_fail=AssertionOnFail.FAIL,
                    ),
                ),
            )
        ]
    )

    assert result.passed is True


def test_thresholds() -> None:
    result = evaluate_thresholds(
        GateConfigSchema(
            pass_rate_min=0.9,
            fail_rate_max=0.1,
            max_pass_rate_delta=0.02,
            max_fail_rate_delta=0.02,
            max_skipped_ratio_delta=0.01,
            max_high_severity_skipped=0,
        ),
        current_metrics=run_metrics(pass_rate=0.88, fail_rate=0.12),
        metrics_delta=metrics_delta(
            pass_rate=-0.03,
            fail_rate=0.01,
            skipped_ratio=0.02,
        ),
        cases=(
            GateCaseResult(
                case_key="skipped_high_case",
                status=CaseResultStatus.SKIPPED,
                case_severity=AssertionSeverity.HIGH,
                assertions=(),
            ),
        ),
    )

    assert result == (
        ThresholdEvaluation(
            criterion=ThresholdCriterion.PASS_RATE_MIN,
            status=ThresholdStatus.FAILED,
            actual=0.88,
            threshold=0.9,
        ),
        ThresholdEvaluation(
            criterion=ThresholdCriterion.FAIL_RATE_MAX,
            status=ThresholdStatus.FAILED,
            actual=0.12,
            threshold=0.1,
        ),
        ThresholdEvaluation(
            criterion=ThresholdCriterion.MAX_PASS_RATE_DELTA,
            status=ThresholdStatus.FAILED,
            actual=0.03,
            threshold=0.02,
        ),
        ThresholdEvaluation(
            criterion=ThresholdCriterion.MAX_FAIL_RATE_DELTA,
            status=ThresholdStatus.PASSED,
            actual=0.01,
            threshold=0.02,
        ),
        ThresholdEvaluation(
            criterion=ThresholdCriterion.MAX_SKIPPED_RATIO_DELTA,
            status=ThresholdStatus.FAILED,
            actual=0.02,
            threshold=0.01,
        ),
        ThresholdEvaluation(
            criterion=ThresholdCriterion.MAX_HIGH_SEVERITY_SKIPPED,
            status=ThresholdStatus.FAILED,
            actual=1,
            threshold=0,
        ),
    )


def test_thresholdsはbaseline不在ならdeltaだけskipする() -> None:
    result = evaluate_thresholds(
        GateConfigSchema(
            pass_rate_min=0.8,
            max_fail_rate_delta=0.05,
        ),
        current_metrics=run_metrics(pass_rate=0.85),
        metrics_delta=None,
    )

    assert result == (
        ThresholdEvaluation(
            criterion=ThresholdCriterion.PASS_RATE_MIN,
            status=ThresholdStatus.PASSED,
            actual=0.85,
            threshold=0.8,
        ),
        ThresholdEvaluation(
            criterion=ThresholdCriterion.MAX_FAIL_RATE_DELTA,
            status=ThresholdStatus.SKIPPED,
            actual=None,
            threshold=0.05,
            reason="baseline_not_available",
        ),
    )


def test_thresholdsは改善deltaを失敗にしない() -> None:
    result = evaluate_thresholds(
        GateConfigSchema(
            max_pass_rate_delta=0.0,
            max_fail_rate_delta=0.0,
            max_skipped_ratio_delta=0.0,
        ),
        current_metrics=run_metrics(),
        metrics_delta=metrics_delta(
            pass_rate=0.01,
            fail_rate=-0.01,
            skipped_ratio=-0.01,
        ),
    )

    assert result == (
        ThresholdEvaluation(
            criterion=ThresholdCriterion.MAX_PASS_RATE_DELTA,
            status=ThresholdStatus.PASSED,
            actual=0.0,
            threshold=0.0,
        ),
        ThresholdEvaluation(
            criterion=ThresholdCriterion.MAX_FAIL_RATE_DELTA,
            status=ThresholdStatus.PASSED,
            actual=0.0,
            threshold=0.0,
        ),
        ThresholdEvaluation(
            criterion=ThresholdCriterion.MAX_SKIPPED_RATIO_DELTA,
            status=ThresholdStatus.PASSED,
            actual=0.0,
            threshold=0.0,
        ),
    )


def test_gate_rules統合は重要失敗でfailedを返す() -> None:
    result = evaluate_gate_rules(
        GateConfigSchema(high_severity_must_pass=True),
        current_metrics=run_metrics(),
        metrics_delta=None,
        cases=(
            GateCaseResult(
                case_key="important_case",
                assertions=(
                    gate_assertion_result(
                        status=AssertionEvaluationStatus.FAILED,
                        severity=AssertionSeverity.HIGH,
                    ),
                ),
            ),
        ),
    )

    assert result.verdict is GateRulesVerdict.FAILED


def test_gate_rules統合はsecret_leak単独でfailedを返す() -> None:
    result = evaluate_gate_rules(
        GateConfigSchema(fail_on_secret_leak=True),
        current_metrics=run_metrics(),
        metrics_delta=None,
        cases=(
            GateCaseResult(
                case_key="reviewed_case",
                final_status="pass",
                assertions=(
                    gate_assertion_result(
                        assertion_id=SECRET_SCAN_ASSERTION_ID,
                        status=AssertionEvaluationStatus.FAILED,
                        severity=AssertionSeverity.HIGH,
                    ),
                ),
            ),
        ),
    )

    assert result.verdict is GateRulesVerdict.FAILED


def test_gate_rules統合は閾値失敗でfailedを返す() -> None:
    result = evaluate_gate_rules(
        GateConfigSchema(pass_rate_min=0.9),
        current_metrics=run_metrics(pass_rate=0.8),
        metrics_delta=None,
    )

    assert result.verdict is GateRulesVerdict.FAILED


def test_gate_rules統合はdelta_skipだけならpassedを返す() -> None:
    result = evaluate_gate_rules(
        GateConfigSchema(max_fail_rate_delta=0.01),
        current_metrics=run_metrics(),
        metrics_delta=None,
    )

    assert result.verdict is GateRulesVerdict.PASSED


def test_gate_rules統合は設定無効ならpassedを返す() -> None:
    result = evaluate_gate_rules(
        GateConfigSchema(),
        current_metrics=run_metrics(pass_rate=0.0, fail_rate=1.0),
        metrics_delta=metrics_delta(fail_rate=1.0),
        cases=(
            GateCaseResult(
                case_key="ignored_case",
                assertions=(
                    gate_assertion_result(
                        assertion_id=SECRET_SCAN_ASSERTION_ID,
                        status=AssertionEvaluationStatus.FAILED,
                        severity=AssertionSeverity.HIGH,
                    ),
                ),
            ),
        ),
    )

    assert result.verdict is GateRulesVerdict.PASSED


def gate_assertion_result(
    *,
    assertion_id: str = "assertion",
    status: AssertionEvaluationStatus = AssertionEvaluationStatus.PASS,
    required: bool = True,
    severity: AssertionSeverity = AssertionSeverity.MEDIUM,
    on_fail: AssertionOnFail = AssertionOnFail.FAIL,
) -> GateAssertionResult:
    return GateAssertionResult(
        assertion_id=assertion_id,
        status=status,
        required=required,
        severity=severity,
        on_fail=on_fail,
    )


def run_metrics(
    *,
    pass_rate: float = 1.0,
    fail_rate: float = 0.0,
    skipped_ratio: float = 0.0,
) -> RunMetrics:
    return RunMetrics(
        total_count=10,
        passed_count=8,
        failed_count=1,
        needs_review_count=0,
        skipped_count=1,
        executed_count=9,
        pass_rate=pass_rate,
        fail_rate=fail_rate,
        skipped_ratio=skipped_ratio,
    )


def metrics_delta(
    *,
    pass_rate: float = 0.0,
    fail_rate: float = 0.0,
    skipped_ratio: float = 0.0,
) -> RunMetricsDelta:
    return RunMetricsDelta(
        total_count=0,
        passed_count=0,
        failed_count=0,
        needs_review_count=0,
        skipped_count=0,
        executed_count=0,
        pass_rate=pass_rate,
        fail_rate=fail_rate,
        skipped_ratio=skipped_ratio,
    )
