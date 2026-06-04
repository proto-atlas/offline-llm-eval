import pytest

from offline_llm_eval.provider.mock import (
    MockProvider,
    MockProviderRequest,
    MockProviderResponseMode,
    invoke_mock_provider,
)
from offline_llm_eval.provider.mock_behavior import (
    InvalidMockResponseError,
    MockErrorResponse,
    MockNormalResponse,
    MockResponseSource,
)
from offline_llm_eval.provider.mock_response_schema import MockCitation
from offline_llm_eval.runner.error_type import ErrorType


def test_invoke_mock_provider_uses_default_response_when_mock_response_is_omitted() -> None:
    result = invoke_mock_provider(MockProviderRequest(expected_answer="Expected answer."))

    assert isinstance(result.response, MockNormalResponse)
    assert result.response.answer == "Expected answer."


def test_invoke_mock_provider_uses_empty_answer_when_expected_answer_is_missing() -> None:
    result = invoke_mock_provider(MockProviderRequest())

    assert isinstance(result.response, MockNormalResponse)
    assert result.response.answer == ""


def test_invoke_mock_provider_marks_default_source() -> None:
    result = invoke_mock_provider(MockProviderRequest())

    assert result.source is MockResponseSource.DEFAULT


def test_invoke_mock_provider_returns_json_mode_for_default_response() -> None:
    result = invoke_mock_provider(MockProviderRequest())

    assert result.response_mode is MockProviderResponseMode.JSON


def test_invoke_mock_provider_adopts_explicit_normal_response() -> None:
    result = invoke_mock_provider(
        MockProviderRequest(
            mock_response={
                "answer": "Use /api/health for readiness.",
                "citations": [
                    {
                        "source": "docs/health",
                        "snippet": "GET /api/health returns readiness.",
                    }
                ],
                "metadata": {"fixture": True},
            }
        )
    )

    assert result.response == MockNormalResponse(
        answer="Use /api/health for readiness.",
        citations=[
            MockCitation(
                source="docs/health",
                snippet="GET /api/health returns readiness.",
            )
        ],
        metadata={"fixture": True},
    )


def test_invoke_mock_provider_marks_explicit_source() -> None:
    result = invoke_mock_provider(
        MockProviderRequest(mock_response={"answer": "ok", "citations": []})
    )

    assert result.source is MockResponseSource.EXPLICIT


def test_invoke_mock_provider_returns_json_mode_for_explicit_response() -> None:
    result = invoke_mock_provider(
        MockProviderRequest(mock_response={"answer": "ok", "citations": []})
    )

    assert result.response_mode is MockProviderResponseMode.JSON


def test_invoke_mock_provider_returns_error_envelope() -> None:
    result = invoke_mock_provider(
        MockProviderRequest(
            mock_response={
                "error": {
                    "code": "overloaded",
                    "message": "providerが過負荷を返しました。",
                }
            }
        )
    )

    assert isinstance(result.response, MockErrorResponse)
    assert result.response.error.code is ErrorType.OVERLOADED


def test_invoke_mock_provider_returns_json_mode_for_error_envelope() -> None:
    result = invoke_mock_provider(
        MockProviderRequest(
            mock_response={
                "error": {
                    "code": "provider_error",
                    "message": "providerが失敗を返しました。",
                }
            }
        )
    )

    assert result.response_mode is MockProviderResponseMode.JSON


def test_invoke_mock_provider_rejects_null_mock_response() -> None:
    with pytest.raises(InvalidMockResponseError) as error:
        invoke_mock_provider(MockProviderRequest(mock_response=None))

    assert error.value.code == "validation_error"


def test_mock_provider_complete_delegates_to_invoke() -> None:
    result = MockProvider().complete(
        MockProviderRequest(mock_response={"answer": "ok", "citations": []})
    )

    assert isinstance(result.response, MockNormalResponse)
    assert result.response.answer == "ok"
