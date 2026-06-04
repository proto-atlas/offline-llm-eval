from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Discriminator, Tag, TypeAdapter, ValidationError

from offline_llm_eval.provider.mock_response_schema import (
    MockNormalResponse as MockNormalResponse,
)
from offline_llm_eval.runner.error_type import ErrorType

MockErrorCode = ErrorType


class MockResponseSource(StrEnum):
    DEFAULT = "default"
    EXPLICIT = "explicit"


class MockErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorType
    message: str


class MockErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: MockErrorEnvelope


def select_mock_response_variant(value: object) -> Hashable | None:
    if isinstance(value, MockNormalResponse):
        return "normal"

    if isinstance(value, MockErrorResponse):
        return "error"

    if isinstance(value, Mapping):
        error_value = value.get("error")
        if isinstance(error_value, Mapping):
            return "error"
        if "error" in value:
            return "error"
        return "normal"

    return None


type MockResponse = Annotated[
    Annotated[MockNormalResponse, Tag("normal")] | Annotated[MockErrorResponse, Tag("error")],
    Discriminator(
        select_mock_response_variant,
        custom_error_message="mock_response は通常応答またはエラー応答である必要があります。",
    ),
]

MOCK_RESPONSE_ADAPTER: Final[TypeAdapter[MockResponse]] = TypeAdapter(MockResponse)


class _MissingMockResponse:
    pass


MISSING_MOCK_RESPONSE: Final = _MissingMockResponse()


@dataclass(frozen=True, slots=True)
class MockResponseBehavior:
    source: MockResponseSource
    response: MockResponse


class InvalidMockResponseError(ValueError):
    def __init__(self, message: str, *, cause: ValidationError | None = None) -> None:
        self.code = "validation_error"
        self.cause = cause
        super().__init__(f"{self.code}: {message}")


def build_default_mock_response(expected_answer: str | None = None) -> MockNormalResponse:
    return MockNormalResponse(answer=expected_answer or "", citations=[], metadata={})


def parse_mock_response(
    value: object = MISSING_MOCK_RESPONSE,
    *,
    expected_answer: str | None = None,
) -> MockResponseBehavior:
    if value is MISSING_MOCK_RESPONSE:
        return MockResponseBehavior(
            source=MockResponseSource.DEFAULT,
            response=build_default_mock_response(expected_answer),
        )

    if value is None:
        raise InvalidMockResponseError("mock_response はnullにできません。")

    try:
        response = MOCK_RESPONSE_ADAPTER.validate_python(value)
    except ValidationError as error:
        raise InvalidMockResponseError("mock_response が不正です。", cause=error) from error

    return MockResponseBehavior(source=MockResponseSource.EXPLICIT, response=response)
