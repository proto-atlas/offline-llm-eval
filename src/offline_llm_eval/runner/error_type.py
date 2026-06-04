from enum import StrEnum
from typing import Final

from pydantic import TypeAdapter, ValidationError


class ErrorType(StrEnum):
    PROVIDER_ERROR = "provider_error"
    OVERLOADED = "overloaded"
    UNKNOWN_ERROR = "unknown_error"
    RESPONSE_MODE_MISMATCH = "response_mode_mismatch"


ERROR_TYPE_VALUES: Final = tuple(error_type.value for error_type in ErrorType)
CASE_RESULT_ERROR_TYPE_CHECK_NAME: Final = "ck_case_results_error_type"
CASE_RESULT_ERROR_TYPE_CHECK_SQL: Final = (
    "error_type is null or error_type in "
    "('provider_error', 'overloaded', 'unknown_error', 'response_mode_mismatch')"
)
ERROR_TYPE_ADAPTER: Final[TypeAdapter[ErrorType]] = TypeAdapter(ErrorType)
OPTIONAL_ERROR_TYPE_ADAPTER: Final[TypeAdapter[ErrorType | None]] = TypeAdapter(ErrorType | None)


class InvalidErrorTypeError(ValueError):
    def __init__(self, message: str, *, cause: ValidationError | None = None) -> None:
        self.code = "validation_error"
        self.cause = cause
        super().__init__(f"{self.code}: {message}")


def parse_error_type(value: object) -> ErrorType:
    try:
        return ERROR_TYPE_ADAPTER.validate_python(value)
    except ValidationError as error:
        raise InvalidErrorTypeError("error_type が不正です。", cause=error) from error


def parse_optional_error_type(value: object) -> ErrorType | None:
    try:
        return OPTIONAL_ERROR_TYPE_ADAPTER.validate_python(value)
    except ValidationError as error:
        raise InvalidErrorTypeError("error_type が不正です。", cause=error) from error
