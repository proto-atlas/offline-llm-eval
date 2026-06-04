from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from offline_llm_eval.dataset.assertion_model import (
    AssertionOnFail,
    AssertionSeverity,
    AssertionType,
)
from offline_llm_eval.evaluator.assertions import AssertionEvaluationStatus
from offline_llm_eval.evaluator.secret_pattern import (
    SECRET_SCAN_ASSERTION_ID,
    SecretScanStatus,
)
from offline_llm_eval.evidence.sanitize import sanitize_evidence_text

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


class UnifiedAssertionType(StrEnum):
    EXACT_MATCH = "exact_match"
    NORMALIZED_CONTAINS = "normalized_contains"
    KEYWORD_ALL = "keyword_all"
    KEYWORD_ANY = "keyword_any"
    REGEX = "regex"
    NO_ANSWER_EXPECTED = "no_answer_expected"
    CITATION_PRESENCE = "citation_presence"
    SOURCE_ID_EXACT_SET = "source_id_exact_set"
    SOURCE_ID_SUBSET = "source_id_subset"
    JSON_SCHEMA = "json_schema"
    FORBIDDEN_PHRASE = "forbidden_phrase"
    LATENCY_THRESHOLD = "latency_threshold"
    SECRET_SCAN = "secret_scan"


class NormalAssertionResultDbSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assertion_db_id: int
    assertion_id: str
    assertion_type: AssertionType
    status: AssertionEvaluationStatus
    detail: str | None
    matched_value: JsonValue
    expected: JsonValue
    required: bool
    severity: AssertionSeverity
    on_fail: AssertionOnFail


class PseudoEvaluatorResultDbSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Literal["__secret_scan__"]
    status: SecretScanStatus
    severity: AssertionSeverity
    required: bool
    on_fail: AssertionOnFail
    detail_code: str | None

    @field_validator("severity")
    @classmethod
    def validate_secret_scan_severity(cls, value: AssertionSeverity) -> AssertionSeverity:
        if value is not AssertionSeverity.HIGH:
            raise ValueError("secret_scan severity は high である必要があります。")
        return value

    @field_validator("required")
    @classmethod
    def validate_secret_scan_required(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("secret_scan required は true である必要があります。")
        return value

    @field_validator("on_fail")
    @classmethod
    def validate_secret_scan_on_fail(cls, value: AssertionOnFail) -> AssertionOnFail:
        if value is not AssertionOnFail.FAIL:
            raise ValueError("secret_scan on_fail は fail である必要があります。")
        return value


class UnifiedEvaluatorResultSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assertion_id: str
    assertion_type: UnifiedAssertionType
    status: AssertionEvaluationStatus
    detail: str | None
    matched_value: JsonValue
    expected: JsonValue
    required: bool
    severity: AssertionSeverity
    on_fail: AssertionOnFail


class InvalidEvaluatorResultSchemaError(ValueError):
    def __init__(self, message: str, *, cause: ValidationError | None = None) -> None:
        self.code = "validation_error"
        self.cause = cause
        super().__init__(f"{self.code}: {message}")


def validate_normal_assertion_result(value: object) -> NormalAssertionResultDbSchema:
    try:
        return NormalAssertionResultDbSchema.model_validate(value)
    except ValidationError as error:
        raise InvalidEvaluatorResultSchemaError(
            "通常の判定結果が不正です。",
            cause=error,
        ) from error


def validate_pseudo_evaluator_result(value: object) -> PseudoEvaluatorResultDbSchema:
    try:
        return PseudoEvaluatorResultDbSchema.model_validate(value)
    except ValidationError as error:
        raise InvalidEvaluatorResultSchemaError(
            "擬似評価結果が不正です。",
            cause=error,
        ) from error


def validate_unified_evaluator_result(value: object) -> UnifiedEvaluatorResultSchema:
    try:
        return UnifiedEvaluatorResultSchema.model_validate(value)
    except ValidationError as error:
        raise InvalidEvaluatorResultSchemaError(
            "統合評価結果が不正です。",
            cause=error,
        ) from error


def normal_result_to_unified(
    result: NormalAssertionResultDbSchema,
) -> UnifiedEvaluatorResultSchema:
    return UnifiedEvaluatorResultSchema(
        assertion_id=result.assertion_id,
        assertion_type=UnifiedAssertionType(result.assertion_type.value),
        status=result.status,
        detail=result.detail,
        matched_value=result.matched_value,
        expected=result.expected,
        required=result.required,
        severity=result.severity,
        on_fail=result.on_fail,
    )


def pseudo_result_to_unified(
    result: PseudoEvaluatorResultDbSchema,
) -> UnifiedEvaluatorResultSchema:
    return UnifiedEvaluatorResultSchema(
        assertion_id=result.id,
        assertion_type=UnifiedAssertionType.SECRET_SCAN,
        status=AssertionEvaluationStatus(result.status.value),
        detail=_sanitize_detail_code(result.detail_code),
        matched_value=None,
        expected=None,
        required=result.required,
        severity=result.severity,
        on_fail=result.on_fail,
    )


def _sanitize_detail_code(detail_code: str | None) -> str | None:
    if detail_code is None:
        return None
    return sanitize_evidence_text(detail_code)


def merge_evaluator_results(
    normal_results: tuple[NormalAssertionResultDbSchema, ...],
    pseudo_results: tuple[PseudoEvaluatorResultDbSchema, ...],
) -> tuple[UnifiedEvaluatorResultSchema, ...]:
    return tuple(
        [*(normal_result_to_unified(result) for result in normal_results)]
        + [*(pseudo_result_to_unified(result) for result in pseudo_results)]
    )


def dump_normal_assertion_result(
    result: NormalAssertionResultDbSchema,
) -> dict[str, JsonValue]:
    return result.model_dump(mode="json")


def dump_pseudo_evaluator_result(
    result: PseudoEvaluatorResultDbSchema,
) -> dict[str, JsonValue]:
    return result.model_dump(mode="json")


def dump_unified_evaluator_result(
    result: UnifiedEvaluatorResultSchema,
) -> dict[str, JsonValue]:
    return result.model_dump(mode="json")


def build_secret_scan_pseudo_result(
    *,
    status: SecretScanStatus,
    detail_code: str | None,
) -> PseudoEvaluatorResultDbSchema:
    return PseudoEvaluatorResultDbSchema(
        id=SECRET_SCAN_ASSERTION_ID,
        status=status,
        severity=AssertionSeverity.HIGH,
        required=True,
        on_fail=AssertionOnFail.FAIL,
        detail_code=detail_code,
    )
