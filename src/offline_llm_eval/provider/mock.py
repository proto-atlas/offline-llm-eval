from dataclasses import dataclass
from enum import StrEnum

from offline_llm_eval.provider.mock_behavior import (
    MISSING_MOCK_RESPONSE,
    MockResponse,
    MockResponseSource,
    parse_mock_response,
)


class MockProviderResponseMode(StrEnum):
    JSON = "json"


@dataclass(frozen=True, slots=True)
class MockProviderRequest:
    mock_response: object = MISSING_MOCK_RESPONSE
    expected_answer: str | None = None


@dataclass(frozen=True, slots=True)
class MockProviderResult:
    response_mode: MockProviderResponseMode
    response: MockResponse
    source: MockResponseSource


class MockProvider:
    response_mode = MockProviderResponseMode.JSON

    def complete(self, request: MockProviderRequest) -> MockProviderResult:
        return invoke_mock_provider(request)


def invoke_mock_provider(request: MockProviderRequest) -> MockProviderResult:
    behavior = parse_mock_response(
        request.mock_response,
        expected_answer=request.expected_answer,
    )
    return MockProviderResult(
        response_mode=MockProviderResponseMode.JSON,
        response=behavior.response,
        source=behavior.source,
    )
