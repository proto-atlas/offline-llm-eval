from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import NamedTuple, TypedDict

from fastapi import status
from fastapi.responses import JSONResponse

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


class ApiErrorCode(StrEnum):
    BASELINE_NOT_FOUND = "baseline_not_found"
    BASELINE_IN_PROGRESS = "baseline_in_progress"
    CONCURRENT_RUN_BLOCKED = "concurrent_run_blocked"
    DATASET_IMPORT_BLOCKED_BY_RUNNING_RUN = "dataset_import_blocked_by_running_run"
    LOCK_TIMEOUT = "lock_timeout"
    VALIDATION_ERROR = "validation_error"
    CASE_NOT_FOUND = "case_not_found"
    RUN_NOT_FOUND = "run_not_found"


class ErrorDefinition(NamedTuple):
    status_code: int
    extra_fields: frozenset[str]


class ApiErrorPayload(TypedDict):
    code: str
    message: str
    extra: dict[str, JsonValue]


class ApiErrorBody(TypedDict):
    error: ApiErrorPayload


class InvalidErrorExtraFieldsError(ValueError):
    pass


ERROR_DEFINITIONS: dict[ApiErrorCode, ErrorDefinition] = {
    ApiErrorCode.BASELINE_NOT_FOUND: ErrorDefinition(
        status.HTTP_404_NOT_FOUND,
        frozenset({"baseline_spec"}),
    ),
    ApiErrorCode.BASELINE_IN_PROGRESS: ErrorDefinition(
        status.HTTP_409_CONFLICT,
        frozenset({"run_id"}),
    ),
    ApiErrorCode.CONCURRENT_RUN_BLOCKED: ErrorDefinition(
        status.HTTP_409_CONFLICT,
        frozenset({"dataset_id", "target_label"}),
    ),
    ApiErrorCode.DATASET_IMPORT_BLOCKED_BY_RUNNING_RUN: ErrorDefinition(
        status.HTTP_409_CONFLICT,
        frozenset({"dataset_id", "running_run_id"}),
    ),
    ApiErrorCode.LOCK_TIMEOUT: ErrorDefinition(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        frozenset(),
    ),
    ApiErrorCode.VALIDATION_ERROR: ErrorDefinition(
        status.HTTP_400_BAD_REQUEST,
        frozenset({"errors"}),
    ),
    ApiErrorCode.CASE_NOT_FOUND: ErrorDefinition(
        status.HTTP_404_NOT_FOUND,
        frozenset({"run_id", "case_key"}),
    ),
    ApiErrorCode.RUN_NOT_FOUND: ErrorDefinition(
        status.HTTP_404_NOT_FOUND,
        frozenset({"run_id"}),
    ),
}


def build_error_body(
    code: ApiErrorCode,
    message: str,
    extra: Mapping[str, JsonValue] | None = None,
) -> ApiErrorBody:
    normalized_extra = dict(extra or {})
    validate_error_extra_fields(code, normalized_extra.keys())
    return {
        "error": {
            "code": code.value,
            "message": message,
            "extra": normalized_extra,
        },
    }


def build_error_response(
    code: ApiErrorCode,
    message: str,
    extra: Mapping[str, JsonValue] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=ERROR_DEFINITIONS[code].status_code,
        content=build_error_body(code, message, extra),
    )


def validate_error_extra_fields(code: ApiErrorCode, actual_fields: Iterable[str]) -> None:
    expected_fields = ERROR_DEFINITIONS[code].extra_fields
    actual_field_set = frozenset(actual_fields)
    if actual_field_set == expected_fields:
        return

    missing_fields = sorted(expected_fields - actual_field_set)
    unexpected_fields = sorted(actual_field_set - expected_fields)
    raise InvalidErrorExtraFieldsError(
        f"{code.value} の extra fields が定義と一致しません: "
        f"missing={missing_fields}, unexpected={unexpected_fields}",
    )
