import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from re import Pattern
from typing import TypeGuard, cast

from offline_llm_eval.dataset.assertion_model import (
    AssertionOnFail,
    AssertionSeverity,
    AssertionType,
)
from offline_llm_eval.evidence.sanitize import sanitize_evidence_text, sanitize_evidence_value

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]

INVALID_EXPECTED_DETAIL = "invalid_expected"


class AssertionEvaluationStatus(StrEnum):
    PASS = "pass"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class AssertionForEvaluation:
    assertion_db_id: int
    assertion_id: str
    assertion_type: AssertionType
    expected: JsonValue = None
    required: bool = True
    severity: AssertionSeverity = AssertionSeverity.MEDIUM
    on_fail: AssertionOnFail = AssertionOnFail.FAIL


@dataclass(frozen=True, slots=True)
class CitationForEvaluation:
    source: str
    snippet: str = ""


@dataclass(frozen=True, slots=True)
class ResponseForEvaluation:
    answer: str
    citations: tuple[CitationForEvaluation, ...] = ()
    metadata: JsonValue = None
    latency_ms: int | float | None = None


@dataclass(frozen=True, slots=True)
class AssertionEvaluation:
    passed: bool | None
    detail: str | None
    matched_value: JsonValue


@dataclass(frozen=True, slots=True)
class EvaluatedAssertionResult:
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


def evaluate_assertions(
    assertions: Sequence[AssertionForEvaluation],
    response: ResponseForEvaluation,
) -> tuple[EvaluatedAssertionResult, ...]:
    return tuple(evaluate_assertion(assertion, response) for assertion in assertions)


def evaluate_assertion(
    assertion: AssertionForEvaluation,
    response: ResponseForEvaluation,
) -> EvaluatedAssertionResult:
    evaluation = _evaluate_assertion(assertion, response)
    return EvaluatedAssertionResult(
        assertion_db_id=assertion.assertion_db_id,
        assertion_id=assertion.assertion_id,
        assertion_type=assertion.assertion_type,
        status=_resolve_status(assertion, evaluation),
        detail=evaluation.detail,
        matched_value=evaluation.matched_value,
        expected=assertion.expected,
        required=assertion.required,
        severity=assertion.severity,
        on_fail=assertion.on_fail,
    )


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _evaluate_assertion(
    assertion: AssertionForEvaluation,
    response: ResponseForEvaluation,
) -> AssertionEvaluation:
    match assertion.assertion_type:
        case AssertionType.EXACT_MATCH:
            return _evaluate_exact_match(assertion.expected, response.answer)
        case AssertionType.NORMALIZED_CONTAINS:
            return _evaluate_normalized_contains(assertion.expected, response.answer)
        case AssertionType.KEYWORD_ALL:
            return _evaluate_keyword_all(assertion.expected, response.answer)
        case AssertionType.KEYWORD_ANY:
            return _evaluate_keyword_any(assertion.expected, response.answer)
        case AssertionType.REGEX:
            return _evaluate_regex(assertion.expected, response.answer)
        case AssertionType.NO_ANSWER_EXPECTED:
            return _evaluate_no_answer_expected(assertion.expected, response.answer)
        case AssertionType.CITATION_PRESENCE:
            return _evaluate_citation_presence(assertion.expected, response.citations)
        case AssertionType.SOURCE_ID_EXACT_SET:
            return _evaluate_source_id_exact_set(assertion.expected, response.citations)
        case AssertionType.SOURCE_ID_SUBSET:
            return _evaluate_source_id_subset(assertion.expected, response.citations)
        case AssertionType.JSON_SCHEMA:
            return _evaluate_json_schema(assertion.expected, response.answer)
        case AssertionType.FORBIDDEN_PHRASE:
            return _evaluate_forbidden_phrase(assertion.expected, response.answer)
        case AssertionType.LATENCY_THRESHOLD:
            return _evaluate_latency_threshold(assertion.expected, response.latency_ms)


def _resolve_status(
    assertion: AssertionForEvaluation,
    evaluation: AssertionEvaluation,
) -> AssertionEvaluationStatus:
    if evaluation.passed is True:
        return AssertionEvaluationStatus.PASS

    if evaluation.passed is None:
        return AssertionEvaluationStatus.SKIPPED

    if assertion.on_fail is AssertionOnFail.WARN or not assertion.required:
        return AssertionEvaluationStatus.WARNING

    return AssertionEvaluationStatus.FAILED


def _evaluate_exact_match(expected: JsonValue, answer: str) -> AssertionEvaluation:
    expected_text = _as_string(expected)
    if expected_text is None:
        return _skipped(INVALID_EXPECTED_DETAIL)

    return _boolean_evaluation(
        normalize_text(answer) == normalize_text(expected_text),
        detail="exact_match_mismatch",
        matched_value=answer,
    )


def _evaluate_normalized_contains(expected: JsonValue, answer: str) -> AssertionEvaluation:
    expected_text = _as_string(expected)
    if expected_text is None:
        return _skipped(INVALID_EXPECTED_DETAIL)

    return _boolean_evaluation(
        normalize_text(expected_text) in normalize_text(answer),
        detail="normalized_contains_missing",
        matched_value=answer,
    )


def _evaluate_keyword_all(expected: JsonValue, answer: str) -> AssertionEvaluation:
    keywords = _as_non_empty_string_tuple(expected)
    if keywords is None:
        return _skipped(INVALID_EXPECTED_DETAIL)

    normalized_answer = normalize_text(answer)
    missing_keywords = tuple(
        keyword for keyword in keywords if normalize_text(keyword) not in normalized_answer
    )
    if missing_keywords:
        return AssertionEvaluation(
            passed=False,
            detail="keyword_all_missing",
            matched_value=cast(JsonValue, list(missing_keywords)),
        )

    return _passed()


def _evaluate_keyword_any(expected: JsonValue, answer: str) -> AssertionEvaluation:
    keywords = _as_non_empty_string_tuple(expected)
    if keywords is None:
        return _skipped(INVALID_EXPECTED_DETAIL)

    normalized_answer = normalize_text(answer)
    matched_keyword = next(
        (keyword for keyword in keywords if normalize_text(keyword) in normalized_answer),
        None,
    )
    if matched_keyword is None:
        return AssertionEvaluation(
            passed=False,
            detail="keyword_any_missing",
            matched_value=None,
        )

    return AssertionEvaluation(passed=True, detail=None, matched_value=matched_keyword)


def _evaluate_regex(expected: JsonValue, answer: str) -> AssertionEvaluation:
    pattern = _compile_pattern(expected)
    if pattern is None:
        return _skipped(INVALID_EXPECTED_DETAIL)

    match = pattern.search(answer)
    if match is None:
        return AssertionEvaluation(
            passed=False,
            detail="regex_no_match",
            matched_value=None,
        )

    return AssertionEvaluation(passed=True, detail=None, matched_value=match.group(0))


def _evaluate_no_answer_expected(expected: JsonValue, answer: str) -> AssertionEvaluation:
    if expected is not True:
        return _skipped(INVALID_EXPECTED_DETAIL)

    normalized_answer = normalize_text(answer)
    no_answer = normalized_answer == "" or normalized_answer in {
        "i do not know",
        "i don't know",
        "not enough information",
        "cannot answer",
        "unknown",
    }
    return _boolean_evaluation(
        no_answer,
        detail="answer_present",
        matched_value=answer,
    )


def _evaluate_citation_presence(
    expected: JsonValue,
    citations: Sequence[CitationForEvaluation],
) -> AssertionEvaluation:
    if not isinstance(expected, bool):
        return _skipped(INVALID_EXPECTED_DETAIL)

    citation_sources = _citation_sources(citations)
    has_citations = bool(citation_sources)
    return _boolean_evaluation(
        has_citations is expected,
        detail="citation_presence_mismatch",
        matched_value=cast(JsonValue, list(citation_sources)),
    )


def _evaluate_source_id_exact_set(
    expected: JsonValue,
    citations: Sequence[CitationForEvaluation],
) -> AssertionEvaluation:
    expected_sources = _as_string_set(expected)
    if expected_sources is None:
        return _skipped(INVALID_EXPECTED_DETAIL)

    actual_sources = set(_citation_sources(citations))
    return _boolean_evaluation(
        actual_sources == expected_sources,
        detail="source_id_exact_set_mismatch",
        matched_value=cast(JsonValue, sorted(actual_sources)),
    )


def _evaluate_source_id_subset(
    expected: JsonValue,
    citations: Sequence[CitationForEvaluation],
) -> AssertionEvaluation:
    expected_sources = _as_string_set(expected)
    if expected_sources is None:
        return _skipped(INVALID_EXPECTED_DETAIL)

    actual_sources = set(_citation_sources(citations))
    return _boolean_evaluation(
        expected_sources.issubset(actual_sources),
        detail="source_id_subset_missing",
        matched_value=cast(JsonValue, sorted(actual_sources)),
    )


def _evaluate_json_schema(expected: JsonValue, answer: str) -> AssertionEvaluation:
    if not isinstance(expected, Mapping):
        return _skipped(INVALID_EXPECTED_DETAIL)

    parsed = _parse_json(answer)
    if parsed is None:
        return _skipped("json_parse_error")

    schema_result = _validate_json_schema_subset(parsed, expected)
    if schema_result == "json_schema_invalid_schema":
        return _skipped(INVALID_EXPECTED_DETAIL)

    return AssertionEvaluation(
        passed=schema_result is None,
        detail=schema_result,
        matched_value=parsed,
    )


def _evaluate_forbidden_phrase(expected: JsonValue, answer: str) -> AssertionEvaluation:
    phrases = _as_non_empty_string_tuple(expected)
    if phrases is None:
        return _skipped(INVALID_EXPECTED_DETAIL)

    normalized_answer = normalize_text(answer)
    matched_phrase = next(
        (phrase for phrase in phrases if normalize_text(phrase) in normalized_answer),
        None,
    )
    if matched_phrase is None:
        return _passed()

    return AssertionEvaluation(
        passed=False,
        detail="forbidden_phrase_present",
        matched_value=matched_phrase,
    )


def _evaluate_latency_threshold(
    expected: JsonValue,
    latency_ms: int | float | None,
) -> AssertionEvaluation:
    threshold = _as_number(expected)
    if threshold is None:
        return _skipped(INVALID_EXPECTED_DETAIL)

    if latency_ms is None:
        return _skipped("latency_missing")

    return _boolean_evaluation(
        latency_ms <= threshold,
        detail="latency_threshold_exceeded",
        matched_value=latency_ms,
    )


def _validate_json_schema_subset(value: JsonValue, schema: Mapping[str, object]) -> str | None:
    expected_type = schema.get("type")
    if expected_type is not None and not isinstance(expected_type, str):
        return "json_schema_invalid_schema"

    if isinstance(expected_type, str) and not _is_supported_json_type(expected_type):
        return "json_schema_invalid_schema"

    if isinstance(expected_type, str) and not _json_type_matches(value, expected_type):
        return "json_schema_type_mismatch"

    if expected_type == "object":
        return _validate_object_schema(value, schema)

    if expected_type == "array":
        return _validate_array_schema(value, schema)

    return None


def _validate_object_schema(value: JsonValue, schema: Mapping[str, object]) -> str | None:
    if not isinstance(value, Mapping):
        return "json_schema_type_mismatch"

    required_error = _validate_required_properties(value, schema.get("required"))
    if required_error is not None:
        return required_error

    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return None

    for property_name, property_schema in properties.items():
        if property_name not in value or not isinstance(property_schema, Mapping):
            continue
        property_error = _validate_json_schema_subset(value[property_name], property_schema)
        if property_error is not None:
            return property_error

    return None


def _validate_required_properties(
    value: Mapping[str, JsonValue],
    required: object,
) -> str | None:
    if required is None:
        return None

    if not _is_string_sequence(required):
        return "json_schema_invalid_schema"

    missing_required = tuple(
        property_name for property_name in required if property_name not in value
    )
    if missing_required:
        return "json_schema_required_missing"

    return None


def _validate_array_schema(value: JsonValue, schema: Mapping[str, object]) -> str | None:
    if not _json_type_matches(value, "array"):
        return "json_schema_type_mismatch"

    items_schema = schema.get("items")
    if items_schema is None:
        return None

    if not isinstance(items_schema, Mapping):
        return "json_schema_invalid_schema"

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return "json_schema_type_mismatch"

    for item in value:
        item_error = _validate_json_schema_subset(item, items_schema)
        if item_error is not None:
            return item_error

    return None


def _sanitize_detail(detail: str | None) -> str | None:
    if detail is None:
        return None
    return sanitize_evidence_text(detail)


def _json_type_matches(value: JsonValue, expected_type: str) -> bool:
    match expected_type:
        case "object":
            return isinstance(value, Mapping)
        case "array":
            return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
        case "string":
            return isinstance(value, str)
        case "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        case "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        case "boolean":
            return isinstance(value, bool)
        case "null":
            return value is None
        case _:
            return False


def _is_supported_json_type(expected_type: str) -> bool:
    return expected_type in {
        "array",
        "boolean",
        "integer",
        "null",
        "number",
        "object",
        "string",
    }


def _parse_json(value: str) -> JsonValue | None:
    try:
        parsed: JsonValue = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed


def _compile_pattern(value: JsonValue) -> Pattern[str] | None:
    pattern = _as_string(value)
    if pattern is None:
        return None

    try:
        return re.compile(pattern)
    except re.error:
        return None


def _as_string(value: JsonValue) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _as_string_tuple(value: JsonValue) -> tuple[str, ...] | None:
    if not _is_string_sequence(value):
        return None
    return tuple(value)


def _as_non_empty_string_tuple(value: JsonValue) -> tuple[str, ...] | None:
    strings = _as_string_tuple(value)
    if strings is None or not strings:
        return None
    return strings


def _as_string_set(value: JsonValue) -> set[str] | None:
    strings = _as_string_tuple(value)
    if strings is None:
        return None
    return set(strings)


def _as_number(value: JsonValue) -> int | float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return value

    return None


def _is_string_sequence(value: object) -> TypeGuard[Sequence[str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    return all(isinstance(item, str) for item in value)


def _citation_sources(citations: Sequence[CitationForEvaluation]) -> tuple[str, ...]:
    return tuple(citation.source for citation in citations)


def _boolean_evaluation(
    passed: bool,
    *,
    detail: str,
    matched_value: JsonValue,
) -> AssertionEvaluation:
    if passed:
        return _passed()

    return AssertionEvaluation(
        passed=False,
        detail=detail,
        matched_value=matched_value,
    )


def _passed() -> AssertionEvaluation:
    return AssertionEvaluation(passed=True, detail=None, matched_value=None)


def _skipped(detail: str) -> AssertionEvaluation:
    return AssertionEvaluation(passed=None, detail=detail, matched_value=None)
