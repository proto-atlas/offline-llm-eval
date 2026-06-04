from offline_llm_eval.dataset.assertion_model import (
    AssertionOnFail,
    AssertionSeverity,
)
from offline_llm_eval.evaluator.assertions import AssertionEvaluationStatus
from offline_llm_eval.evaluator.results_schema import (
    UnifiedAssertionType,
    UnifiedEvaluatorResultSchema,
)
from offline_llm_eval.evaluator.status import (
    CaseResultStatus,
    resolve_case_status,
)
from offline_llm_eval.runner.error_type import ErrorType


def normal_result(
    *,
    status: AssertionEvaluationStatus,
    required: bool = True,
    severity: AssertionSeverity = AssertionSeverity.MEDIUM,
    on_fail: AssertionOnFail = AssertionOnFail.FAIL,
) -> UnifiedEvaluatorResultSchema:
    return UnifiedEvaluatorResultSchema(
        assertion_id="answer_exact",
        assertion_type=UnifiedAssertionType.EXACT_MATCH,
        status=status,
        detail=None,
        matched_value=None,
        expected="expected",
        required=required,
        severity=severity,
        on_fail=on_fail,
    )


def secret_scan_result(
    status: AssertionEvaluationStatus = AssertionEvaluationStatus.PASS,
) -> UnifiedEvaluatorResultSchema:
    return UnifiedEvaluatorResultSchema(
        assertion_id="__secret_scan__",
        assertion_type=UnifiedAssertionType.SECRET_SCAN,
        status=status,
        detail=None,
        matched_value=None,
        expected=None,
        required=True,
        severity=AssertionSeverity.HIGH,
        on_fail=AssertionOnFail.FAIL,
    )


def test_needs_review失敗はpriority1で最優先になる() -> None:
    decision = resolve_case_status(
        case_severity=AssertionSeverity.HIGH,
        evaluator_results=(
            normal_result(
                status=AssertionEvaluationStatus.FAILED,
                on_fail=AssertionOnFail.NEEDS_REVIEW,
            ),
            normal_result(status=AssertionEvaluationStatus.FAILED),
            secret_scan_result(status=AssertionEvaluationStatus.FAILED),
        ),
    )

    assert decision.status is CaseResultStatus.NEEDS_REVIEW
    assert decision.priority == 1
    assert decision.detail == "needs_review_failure"


def test_high_caseのrequired_fail失敗はpriority2でfailedになる() -> None:
    decision = resolve_case_status(
        case_severity=AssertionSeverity.HIGH,
        evaluator_results=(
            normal_result(status=AssertionEvaluationStatus.FAILED),
            secret_scan_result(),
        ),
    )

    assert decision.status is CaseResultStatus.FAILED
    assert decision.priority == 2
    assert decision.detail == "high_case_required_failure"


def test_medium_caseのrequired_fail失敗はpriority3でfailedになる() -> None:
    decision = resolve_case_status(
        case_severity=AssertionSeverity.MEDIUM,
        evaluator_results=(
            normal_result(status=AssertionEvaluationStatus.FAILED),
            secret_scan_result(),
        ),
    )

    assert decision.status is CaseResultStatus.FAILED
    assert decision.priority == 3
    assert decision.detail == "required_failure"


def test_secret_scan失敗はcase_severityに関わらずfailedになる() -> None:
    decision = resolve_case_status(
        case_severity=AssertionSeverity.LOW,
        evaluator_results=(
            normal_result(status=AssertionEvaluationStatus.PASS),
            secret_scan_result(status=AssertionEvaluationStatus.FAILED),
        ),
    )

    assert decision.status is CaseResultStatus.FAILED
    assert decision.priority == 3


def test通常assertionが全skippedならpriority4でskippedになる() -> None:
    decision = resolve_case_status(
        case_severity=AssertionSeverity.MEDIUM,
        evaluator_results=(
            normal_result(status=AssertionEvaluationStatus.SKIPPED),
            normal_result(status=AssertionEvaluationStatus.NOT_APPLICABLE),
            secret_scan_result(),
        ),
    )

    assert decision.status is CaseResultStatus.SKIPPED
    assert decision.priority == 4
    assert decision.detail == "all_normal_assertions_skipped_or_case_error"


def test通常assertion空でもerror_typeありならpriority4でskippedになる() -> None:
    decision = resolve_case_status(
        case_severity=AssertionSeverity.MEDIUM,
        evaluator_results=(secret_scan_result(),),
        case_error_type=ErrorType.PROVIDER_ERROR.value,
    )

    assert decision.status is CaseResultStatus.SKIPPED
    assert decision.priority == 4


def testresponse_mode_mismatchでsecret_scan_not_applicableでもpriority4になる() -> None:
    decision = resolve_case_status(
        case_severity=AssertionSeverity.MEDIUM,
        evaluator_results=(secret_scan_result(status=AssertionEvaluationStatus.NOT_APPLICABLE),),
        case_error_type=ErrorType.RESPONSE_MODE_MISMATCH.value,
    )

    assert decision.status is CaseResultStatus.SKIPPED
    assert decision.priority == 4


def testwarningのみならpriority5でpassになる() -> None:
    decision = resolve_case_status(
        case_severity=AssertionSeverity.MEDIUM,
        evaluator_results=(
            normal_result(
                status=AssertionEvaluationStatus.WARNING,
                on_fail=AssertionOnFail.WARN,
            ),
            normal_result(
                status=AssertionEvaluationStatus.WARNING,
                required=False,
            ),
            secret_scan_result(),
        ),
    )

    assert decision.status is CaseResultStatus.PASS
    assert decision.priority == 5
    assert decision.detail == "warning_only"


def testwarningのみでsecret_scan_not_applicableならpriority5でpassになる() -> None:
    decision = resolve_case_status(
        case_severity=AssertionSeverity.MEDIUM,
        evaluator_results=(
            normal_result(
                status=AssertionEvaluationStatus.WARNING,
                on_fail=AssertionOnFail.WARN,
            ),
            secret_scan_result(status=AssertionEvaluationStatus.NOT_APPLICABLE),
        ),
    )

    assert decision.status is CaseResultStatus.PASS
    assert decision.priority == 5
    assert decision.detail == "warning_only"


def testpassとwarning混在ならpriority6でpassになる() -> None:
    decision = resolve_case_status(
        case_severity=AssertionSeverity.MEDIUM,
        evaluator_results=(
            normal_result(status=AssertionEvaluationStatus.PASS),
            normal_result(
                status=AssertionEvaluationStatus.WARNING,
                required=False,
            ),
            secret_scan_result(),
        ),
    )

    assert decision.status is CaseResultStatus.PASS
    assert decision.priority == 6
    assert decision.detail == "pass"


def test全passならpriority6でpassになる() -> None:
    decision = resolve_case_status(
        case_severity=AssertionSeverity.MEDIUM,
        evaluator_results=(
            normal_result(status=AssertionEvaluationStatus.PASS),
            secret_scan_result(),
        ),
    )

    assert decision.status is CaseResultStatus.PASS
    assert decision.priority == 6


def test通常assertion空かつerror_typeなしならvacuously_trueでpriority6になる() -> None:
    decision = resolve_case_status(
        case_severity=AssertionSeverity.MEDIUM,
        evaluator_results=(secret_scan_result(),),
    )

    assert decision.status is CaseResultStatus.PASS
    assert decision.priority == 6
