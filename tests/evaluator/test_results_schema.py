import pytest

from offline_llm_eval.dataset.assertion_model import (
    AssertionOnFail,
    AssertionSeverity,
    AssertionType,
)
from offline_llm_eval.evaluator.assertions import AssertionEvaluationStatus
from offline_llm_eval.evaluator.results_schema import (
    InvalidEvaluatorResultSchemaError,
    NormalAssertionResultDbSchema,
    PseudoEvaluatorResultDbSchema,
    UnifiedAssertionType,
    build_secret_scan_pseudo_result,
    dump_normal_assertion_result,
    dump_pseudo_evaluator_result,
    dump_unified_evaluator_result,
    merge_evaluator_results,
    normal_result_to_unified,
    pseudo_result_to_unified,
    validate_normal_assertion_result,
    validate_pseudo_evaluator_result,
    validate_unified_evaluator_result,
)
from offline_llm_eval.evaluator.secret_pattern import (
    SECRET_SCAN_ASSERTION_ID,
    SecretScanStatus,
)


def normal_payload() -> dict[str, object]:
    return {
        "assertion_db_id": 10,
        "assertion_id": "answer_exact",
        "assertion_type": "exact_match",
        "status": "failed",
        "detail": "exact_match_mismatch",
        "matched_value": "actual",
        "expected": "expected",
        "required": True,
        "severity": "high",
        "on_fail": "fail",
    }


def pseudo_payload() -> dict[str, object]:
    return {
        "id": SECRET_SCAN_ASSERTION_ID,
        "status": "failed",
        "severity": "high",
        "required": True,
        "on_fail": "fail",
        "detail_code": "aws_access_key",
    }


def test_normal_assertion_result_通常DB形を検証できる() -> None:
    result = validate_normal_assertion_result(normal_payload())

    assert result == NormalAssertionResultDbSchema(
        assertion_db_id=10,
        assertion_id="answer_exact",
        assertion_type=AssertionType.EXACT_MATCH,
        status=AssertionEvaluationStatus.FAILED,
        detail="exact_match_mismatch",
        matched_value="actual",
        expected="expected",
        required=True,
        severity=AssertionSeverity.HIGH,
        on_fail=AssertionOnFail.FAIL,
    )


def test_normal_assertion_result_extra_fieldを拒否する() -> None:
    payload = normal_payload()
    payload["extra"] = True

    with pytest.raises(InvalidEvaluatorResultSchemaError) as error:
        validate_normal_assertion_result(payload)

    assert error.value.code == "validation_error"


def test_normal_assertion_result_secret_scan型は通常DBで拒否する() -> None:
    payload = normal_payload()
    payload["assertion_type"] = "secret_scan"

    with pytest.raises(InvalidEvaluatorResultSchemaError):
        validate_normal_assertion_result(payload)


def test_pseudo_evaluator_result_6fieldを検証できる() -> None:
    result = validate_pseudo_evaluator_result(pseudo_payload())

    assert result == PseudoEvaluatorResultDbSchema(
        id=SECRET_SCAN_ASSERTION_ID,
        status=SecretScanStatus.FAILED,
        severity=AssertionSeverity.HIGH,
        required=True,
        on_fail=AssertionOnFail.FAIL,
        detail_code="aws_access_key",
    )


def test_pseudo_evaluator_result_id固定値以外を拒否する() -> None:
    payload = pseudo_payload()
    payload["id"] = "not_secret_scan"

    with pytest.raises(InvalidEvaluatorResultSchemaError):
        validate_pseudo_evaluator_result(payload)


def test_pseudo_evaluator_result_high以外のseverityを拒否する() -> None:
    payload = pseudo_payload()
    payload["severity"] = "medium"

    with pytest.raises(InvalidEvaluatorResultSchemaError):
        validate_pseudo_evaluator_result(payload)


def test_build_secret_scan_pseudo_result_固定fieldを補う() -> None:
    result = build_secret_scan_pseudo_result(
        status=SecretScanStatus.NOT_APPLICABLE,
        detail_code="no_response_to_scan",
    )

    assert result.id == SECRET_SCAN_ASSERTION_ID
    assert result.severity is AssertionSeverity.HIGH
    assert result.required is True
    assert result.on_fail is AssertionOnFail.FAIL


def test_unified_evaluator_result_通常12種とsecret_scanを検証できる() -> None:
    normal = validate_unified_evaluator_result(
        {
            "assertion_id": "answer_exact",
            "assertion_type": "exact_match",
            "status": "pass",
            "detail": None,
            "matched_value": None,
            "expected": "expected",
            "required": True,
            "severity": "medium",
            "on_fail": "fail",
        }
    )
    pseudo = validate_unified_evaluator_result(
        {
            "assertion_id": SECRET_SCAN_ASSERTION_ID,
            "assertion_type": "secret_scan",
            "status": "failed",
            "detail": "aws_access_key",
            "matched_value": None,
            "expected": None,
            "required": True,
            "severity": "high",
            "on_fail": "fail",
        }
    )

    assert normal.assertion_type is UnifiedAssertionType.EXACT_MATCH
    assert pseudo.assertion_type is UnifiedAssertionType.SECRET_SCAN


def test_normal_result_to_unified_assertion_db_idを落として9fieldにする() -> None:
    result = normal_result_to_unified(validate_normal_assertion_result(normal_payload()))

    assert dump_unified_evaluator_result(result) == {
        "assertion_id": "answer_exact",
        "assertion_type": "exact_match",
        "status": "failed",
        "detail": "exact_match_mismatch",
        "matched_value": "actual",
        "expected": "expected",
        "required": True,
        "severity": "high",
        "on_fail": "fail",
    }


def test_pseudo_result_to_unified_paddingで9fieldにする() -> None:
    result = pseudo_result_to_unified(validate_pseudo_evaluator_result(pseudo_payload()))

    assert dump_unified_evaluator_result(result) == {
        "assertion_id": SECRET_SCAN_ASSERTION_ID,
        "assertion_type": "secret_scan",
        "status": "failed",
        "detail": "aws_access_key",
        "matched_value": None,
        "expected": None,
        "required": True,
        "severity": "high",
        "on_fail": "fail",
    }


def test_pseudo_result_to_unifiedはdetail_codeのsecret形状値をmaskする() -> None:
    payload = pseudo_payload()
    payload["detail_code"] = "".join(("AKIA", "12345678", "90ABCDEF"))

    result = pseudo_result_to_unified(validate_pseudo_evaluator_result(payload))

    assert result.detail == "[masked:aws_access_key]"


def test_merge_evaluator_results_通常結果の後にpseudoを追加する() -> None:
    normal = validate_normal_assertion_result(normal_payload())
    pseudo = validate_pseudo_evaluator_result(pseudo_payload())

    results = merge_evaluator_results((normal,), (pseudo,))

    assert [result.assertion_id for result in results] == [
        "answer_exact",
        SECRET_SCAN_ASSERTION_ID,
    ]


def test_dump_normal_assertion_result_enumをjson値にする() -> None:
    result = validate_normal_assertion_result(normal_payload())

    assert dump_normal_assertion_result(result)["assertion_type"] == "exact_match"
    assert dump_normal_assertion_result(result)["status"] == "failed"


def test_dump_pseudo_evaluator_result_enumをjson値にする() -> None:
    result = validate_pseudo_evaluator_result(pseudo_payload())

    assert dump_pseudo_evaluator_result(result) == pseudo_payload()
