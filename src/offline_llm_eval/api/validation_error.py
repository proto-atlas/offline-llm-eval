from collections.abc import Mapping, Sequence
from typing import Final, cast

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from offline_llm_eval.dataset.repository import JsonValue
from offline_llm_eval.evidence.sanitize import sanitize_evidence_text

INPUT_FIELD = "input"
REQUEST_VALIDATION_STATUS_CODE: Final = 422


async def request_validation_error_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=REQUEST_VALIDATION_STATUS_CODE,
        content={"detail": sanitize_validation_errors(exc.errors())},
    )


def sanitize_validation_errors(errors: Sequence[object]) -> list[JsonValue]:
    return [_sanitize_validation_error_value(error) for error in errors]


def _sanitize_validation_error_value(value: object) -> JsonValue:
    if value is None:
        return None

    if isinstance(value, str):
        return sanitize_evidence_text(value)

    if isinstance(value, bool | int | float):
        return value

    if isinstance(value, Mapping):
        return _sanitize_validation_error_mapping(cast(Mapping[object, object], value))

    if isinstance(value, Sequence):
        return [_sanitize_validation_error_value(item) for item in cast(Sequence[object], value)]

    return sanitize_evidence_text(str(value))


def _sanitize_validation_error_mapping(
    value: Mapping[object, object],
) -> dict[str, JsonValue]:
    sanitized: dict[str, JsonValue] = {}
    for key, item in value.items():
        normalized_key = str(key)
        if normalized_key == INPUT_FIELD:
            continue
        sanitized[sanitize_evidence_text(normalized_key)] = _sanitize_validation_error_value(item)
    return sanitized
