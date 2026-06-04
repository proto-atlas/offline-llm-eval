from pydantic import BaseModel, ConfigDict, Field, ValidationError

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


class MockCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    snippet: str


class MockNormalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[MockCitation]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class InvalidMockProviderResponseError(ValueError):
    def __init__(self, message: str, *, cause: ValidationError | None = None) -> None:
        self.code = "validation_error"
        self.cause = cause
        super().__init__(f"{self.code}: {message}")


def validate_mock_provider_response(value: object) -> MockNormalResponse:
    try:
        return MockNormalResponse.model_validate(value)
    except ValidationError as error:
        raise InvalidMockProviderResponseError(
            "mock provider response が不正です。",
            cause=error,
        ) from error


def dump_mock_provider_response(response: MockNormalResponse) -> dict[str, JsonValue]:
    return response.model_dump(mode="json")
