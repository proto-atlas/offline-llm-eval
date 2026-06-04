import pytest

from offline_llm_eval.dataset.assertion_model import (
    AssertionOnFail,
    AssertionSeverity,
    AssertionType,
)
from offline_llm_eval.evaluator.secret_pattern import (
    SECRET_SCAN_ASSERTION_ID,
    SecretScanResult,
    SecretScanStatus,
    build_secret_scan_input,
)
from offline_llm_eval.provider.mock_behavior import (
    MockErrorCode,
    MockErrorEnvelope,
    MockErrorResponse,
)
from offline_llm_eval.runner.error_route import (
    NO_RESPONSE_TO_SCAN_DETAIL,
    AssertionForRoute,
    AssertionResultStatus,
    CaseResultStatus,
    InvalidSecretScanRouteError,
    RoutedAssertionResult,
    route_mock_error_envelope,
    route_response_mode_mismatch,
)


def test_mock_error_envelope_marks_regular_assertions_not_applicable() -> None:
    result = route_mock_error_envelope(
        _error_response(MockErrorCode.PROVIDER_ERROR),
        (_assertion(),),
        _passing_secret_scan,
    )

    assert result.assertion_results[0].status is AssertionResultStatus.NOT_APPLICABLE


def test_mock_error_envelope_keeps_assertion_snapshot_fields() -> None:
    result = route_mock_error_envelope(
        _error_response(MockErrorCode.PROVIDER_ERROR),
        (
            _assertion(
                expected="expected answer",
                required=False,
                severity=AssertionSeverity.LOW,
                on_fail=AssertionOnFail.WARN,
            ),
        ),
        _passing_secret_scan,
    )

    assert result.assertion_results[0].to_db_payload() == {
        "assertion_db_id": 101,
        "assertion_id": "answer_contains",
        "assertion_type": "normalized_contains",
        "status": "not_applicable",
        "detail": None,
        "matched_value": None,
        "expected": "expected answer",
        "required": False,
        "severity": "low",
        "on_fail": "warn",
    }


def test_mock_error_envelope_masks_secret_in_assertion_payload() -> None:
    secret_value = "".join(("AKIA", "IOSFODNN7EXAMPLE"))
    result = route_mock_error_envelope(
        _error_response(MockErrorCode.PROVIDER_ERROR),
        (_assertion(expected=secret_value),),
        _passing_secret_scan,
    )

    assert result.assertion_results[0].to_db_payload()["expected"] == ("[masked:aws_access_key]")


def test_error_route_assertion_payloadはdetailのsecret形状値をmaskする() -> None:
    secret_value = "".join(("AKIA", "90123456", "7890ABCD"))
    routed_result = RoutedAssertionResult(
        assertion_db_id=101,
        assertion_id="answer_contains",
        assertion_type=AssertionType.NORMALIZED_CONTAINS,
        status=AssertionResultStatus.NOT_APPLICABLE,
        detail=f"detail {secret_value}",
        matched_value=None,
        expected=None,
        required=True,
        severity=AssertionSeverity.MEDIUM,
        on_fail=AssertionOnFail.FAIL,
    )

    assert routed_result.to_db_payload()["detail"] == "detail [masked:aws_access_key]"


def test_error_route_assertion_payloadはdict_keyのsecret形状値をmaskする() -> None:
    secret_value = "".join(("AKIA", "70123456", "7890ABCD"))
    routed_result = RoutedAssertionResult(
        assertion_db_id=101,
        assertion_id="answer_contains",
        assertion_type=AssertionType.NORMALIZED_CONTAINS,
        status=AssertionResultStatus.NOT_APPLICABLE,
        detail=None,
        matched_value={secret_value: "actual"},
        expected={"nested": {secret_value: "expected"}},
        required=True,
        severity=AssertionSeverity.MEDIUM,
        on_fail=AssertionOnFail.FAIL,
    )

    payload = routed_result.to_db_payload()

    assert payload["matched_value"] == {"[masked:aws_access_key]": "actual"}
    assert payload["expected"] == {"nested": {"[masked:aws_access_key]": "expected"}}
    assert secret_value not in str(payload)


def test_mock_error_envelope_runs_secret_scan_against_error_strings() -> None:
    access_key = "".join(("AKIA", "IOSFODNN7EXAMPLE"))
    result = route_mock_error_envelope(
        _error_response(
            MockErrorCode.PROVIDER_ERROR,
            message=f"providerが {access_key} を返しました。",
        ),
        (),
        lambda text: SecretScanResult(
            status=SecretScanStatus.FAILED,
            detail_code=text,
        ),
    )

    assert result.evaluator_results_json[0].detail_code == (
        f"provider_error\nproviderが {access_key} を返しました。"
    )


def test_mock_error_envelope_secret_scan_pass_skips_case() -> None:
    result = route_mock_error_envelope(
        _error_response(MockErrorCode.OVERLOADED),
        (_assertion(),),
        _passing_secret_scan,
    )

    assert result.case_status is CaseResultStatus.SKIPPED


def test_mock_error_envelope_records_error_type_from_error_code() -> None:
    result = route_mock_error_envelope(
        _error_response(MockErrorCode.UNKNOWN_ERROR),
        (_assertion(),),
        _passing_secret_scan,
    )

    assert result.error_type is MockErrorCode.UNKNOWN_ERROR


def test_mock_error_envelope_secret_scan_failed_fails_case() -> None:
    result = route_mock_error_envelope(
        _error_response(MockErrorCode.PROVIDER_ERROR),
        (_assertion(),),
        lambda text: SecretScanResult(
            status=SecretScanStatus.FAILED,
            detail_code="aws_access_key",
        ),
    )

    assert result.case_status is CaseResultStatus.FAILED


def test_mock_error_envelope_adds_secret_scan_pseudo_result() -> None:
    result = route_mock_error_envelope(
        _error_response(MockErrorCode.PROVIDER_ERROR),
        (),
        _passing_secret_scan,
    )

    assert result.evaluator_results_json[0].to_json_payload() == {
        "id": SECRET_SCAN_ASSERTION_ID,
        "status": "pass",
        "severity": "high",
        "required": True,
        "on_fail": "fail",
        "detail_code": None,
    }


def test_mock_error_envelope_response_mode_mismatch_code_still_runs_secret_scan() -> None:
    result = route_mock_error_envelope(
        _error_response(MockErrorCode.RESPONSE_MODE_MISMATCH),
        (),
        _passing_secret_scan,
    )

    assert result.evaluator_results_json[0].status is SecretScanStatus.PASS


def test_mock_error_envelope_with_no_assertions_still_skips_case_when_secret_scan_passes() -> None:
    result = route_mock_error_envelope(
        _error_response(MockErrorCode.PROVIDER_ERROR),
        (),
        _passing_secret_scan,
    )

    assert result.case_status is CaseResultStatus.SKIPPED


def test_mock_error_envelope_rejects_not_applicable_secret_scan() -> None:
    with pytest.raises(InvalidSecretScanRouteError) as error:
        route_mock_error_envelope(
            _error_response(MockErrorCode.PROVIDER_ERROR),
            (),
            lambda text: SecretScanResult(status=SecretScanStatus.NOT_APPLICABLE),
        )

    assert error.value.code == "invalid_secret_scan_route"


def test_response_mode_mismatch_marks_regular_assertions_not_applicable() -> None:
    result = route_response_mode_mismatch((_assertion(),))

    assert result.assertion_results[0].status is AssertionResultStatus.NOT_APPLICABLE


def test_response_mode_mismatch_marks_secret_scan_not_applicable() -> None:
    result = route_response_mode_mismatch((_assertion(),))

    assert result.evaluator_results_json[0].to_json_payload() == {
        "id": SECRET_SCAN_ASSERTION_ID,
        "status": "not_applicable",
        "severity": "high",
        "required": True,
        "on_fail": "fail",
        "detail_code": NO_RESPONSE_TO_SCAN_DETAIL,
    }


def test_response_mode_mismatch_records_error_type() -> None:
    result = route_response_mode_mismatch((_assertion(),))

    assert result.error_type is MockErrorCode.RESPONSE_MODE_MISMATCH


def test_response_mode_mismatch_skips_case_with_assertions() -> None:
    result = route_response_mode_mismatch((_assertion(),))

    assert result.case_status is CaseResultStatus.SKIPPED


def test_response_mode_mismatch_skips_case_without_assertions() -> None:
    result = route_response_mode_mismatch(())

    assert result.case_status is CaseResultStatus.SKIPPED


def test_secret_scan_input_collects_nested_error_strings() -> None:
    assert (
        build_secret_scan_input(
            _error_response(MockErrorCode.PROVIDER_ERROR, message="providerが失敗を返しました。")
        )
        == "provider_error\nproviderが失敗を返しました。"
    )


def _assertion(
    *,
    expected: str | None = None,
    required: bool = True,
    severity: AssertionSeverity = AssertionSeverity.MEDIUM,
    on_fail: AssertionOnFail = AssertionOnFail.FAIL,
) -> AssertionForRoute:
    return AssertionForRoute(
        assertion_db_id=101,
        assertion_id="answer_contains",
        assertion_type=AssertionType.NORMALIZED_CONTAINS,
        expected=expected,
        required=required,
        severity=severity,
        on_fail=on_fail,
    )


def _error_response(
    code: MockErrorCode,
    *,
    message: str = "providerが失敗を返しました。",
) -> MockErrorResponse:
    return MockErrorResponse(error=MockErrorEnvelope(code=code, message=message))


def _passing_secret_scan(text: str) -> SecretScanResult:
    return SecretScanResult(status=SecretScanStatus.PASS)
