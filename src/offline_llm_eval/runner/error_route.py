from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

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
from offline_llm_eval.evidence.sanitize import sanitize_evidence_text, sanitize_evidence_value
from offline_llm_eval.provider.mock_behavior import MockErrorResponse
from offline_llm_eval.runner.error_type import ErrorType

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type SecretScanner = Callable[[str], SecretScanResult]

NO_RESPONSE_TO_SCAN_DETAIL: Final = "no_response_to_scan"


class CaseResultStatus(StrEnum):
    FAILED = "failed"
    SKIPPED = "skipped"


class AssertionResultStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"


class InvalidSecretScanRouteError(ValueError):
    def __init__(self, message: str) -> None:
        self.code = "invalid_secret_scan_route"
        super().__init__(f"{self.code}: {message}")


@dataclass(frozen=True, slots=True)
class AssertionForRoute:
    assertion_db_id: int
    assertion_id: str
    assertion_type: AssertionType
    expected: JsonValue = None
    required: bool = True
    severity: AssertionSeverity = AssertionSeverity.MEDIUM
    on_fail: AssertionOnFail = AssertionOnFail.FAIL


@dataclass(frozen=True, slots=True)
class RoutedAssertionResult:
    assertion_db_id: int
    assertion_id: str
    assertion_type: AssertionType
    status: AssertionResultStatus
    detail: str | None
    matched_value: JsonValue
    expected: JsonValue
    required: bool
    severity: AssertionSeverity
    on_fail: AssertionOnFail

    def to_db_payload(self) -> dict[str, JsonValue]:
        return {
            "assertion_db_id": self.assertion_db_id,
            "assertion_id": self.assertion_id,
            "assertion_type": self.assertion_type,
            "status": self.status,
            "detail": _sanitize_detail(self.detail),
            "matched_value": sanitize_evidence_value(self.matched_value),
            "expected": sanitize_evidence_value(self.expected),
            "required": self.required,
            "severity": self.severity,
            "on_fail": self.on_fail,
        }


def _sanitize_detail(detail: str | None) -> str | None:
    if detail is None:
        return None
    return sanitize_evidence_text(detail)


@dataclass(frozen=True, slots=True)
class SecretScanEvaluatorResult:
    id: str
    status: SecretScanStatus
    severity: AssertionSeverity
    required: bool
    on_fail: AssertionOnFail
    detail_code: str | None

    def to_json_payload(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "status": self.status,
            "severity": self.severity,
            "required": self.required,
            "on_fail": self.on_fail,
            "detail_code": self.detail_code,
        }


@dataclass(frozen=True, slots=True)
class ErrorRouteResult:
    case_status: CaseResultStatus
    error_type: ErrorType
    assertion_results: tuple[RoutedAssertionResult, ...]
    evaluator_results_json: tuple[SecretScanEvaluatorResult, ...]


def route_mock_error_envelope(
    response: MockErrorResponse,
    assertions: Sequence[AssertionForRoute],
    scan_secret: SecretScanner,
) -> ErrorRouteResult:
    secret_scan_result = scan_secret(build_secret_scan_input(response))
    if secret_scan_result.status is SecretScanStatus.NOT_APPLICABLE:
        raise InvalidSecretScanRouteError(
            "mock error envelope の secret_scan は pass または failed である必要があります。",
        )

    return ErrorRouteResult(
        case_status=_resolve_mock_error_case_status(secret_scan_result),
        error_type=response.error.code,
        assertion_results=build_not_applicable_assertion_results(assertions),
        evaluator_results_json=(
            SecretScanEvaluatorResult(
                id=SECRET_SCAN_ASSERTION_ID,
                status=secret_scan_result.status,
                severity=AssertionSeverity.HIGH,
                required=True,
                on_fail=AssertionOnFail.FAIL,
                detail_code=secret_scan_result.detail_code,
            ),
        ),
    )


def route_response_mode_mismatch(
    assertions: Sequence[AssertionForRoute],
) -> ErrorRouteResult:
    return ErrorRouteResult(
        case_status=CaseResultStatus.SKIPPED,
        error_type=ErrorType.RESPONSE_MODE_MISMATCH,
        assertion_results=build_not_applicable_assertion_results(assertions),
        evaluator_results_json=(
            SecretScanEvaluatorResult(
                id=SECRET_SCAN_ASSERTION_ID,
                status=SecretScanStatus.NOT_APPLICABLE,
                severity=AssertionSeverity.HIGH,
                required=True,
                on_fail=AssertionOnFail.FAIL,
                detail_code=NO_RESPONSE_TO_SCAN_DETAIL,
            ),
        ),
    )


def build_not_applicable_assertion_results(
    assertions: Sequence[AssertionForRoute],
) -> tuple[RoutedAssertionResult, ...]:
    return tuple(
        RoutedAssertionResult(
            assertion_db_id=assertion.assertion_db_id,
            assertion_id=assertion.assertion_id,
            assertion_type=assertion.assertion_type,
            status=AssertionResultStatus.NOT_APPLICABLE,
            detail=None,
            matched_value=None,
            expected=assertion.expected,
            required=assertion.required,
            severity=assertion.severity,
            on_fail=assertion.on_fail,
        )
        for assertion in assertions
    )


def _resolve_mock_error_case_status(secret_scan_result: SecretScanResult) -> CaseResultStatus:
    if secret_scan_result.status is SecretScanStatus.FAILED:
        return CaseResultStatus.FAILED
    return CaseResultStatus.SKIPPED
