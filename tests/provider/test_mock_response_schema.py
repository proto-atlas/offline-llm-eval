import pytest

from offline_llm_eval.provider.mock_response_schema import (
    InvalidMockProviderResponseError,
    MockCitation,
    MockNormalResponse,
    dump_mock_provider_response,
    validate_mock_provider_response,
)


def test_validate_mock_provider_response_accepts_answer_citations_and_metadata() -> None:
    response = validate_mock_provider_response(
        {
            "answer": "Use /health for liveness.",
            "citations": [{"source": "docs/health", "snippet": "GET /health returns ok."}],
            "metadata": {"nested": {"count": 1, "labels": ["health", "api"]}},
        }
    )

    assert response == MockNormalResponse(
        answer="Use /health for liveness.",
        citations=[MockCitation(source="docs/health", snippet="GET /health returns ok.")],
        metadata={"nested": {"count": 1, "labels": ["health", "api"]}},
    )


def test_validate_mock_provider_response_defaults_metadata_to_empty_dict() -> None:
    response = validate_mock_provider_response(
        {
            "answer": "ok",
            "citations": [],
        }
    )

    assert response.metadata == {}


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"answer": "ok"},
        {"answer": "ok", "citations": [{"source": "docs"}]},
        {"answer": "ok", "citations": [], "extra": True},
        {"answer": "ok", "citations": [{"source": "docs", "snippet": "x", "extra": True}]},
        {"answer": "ok", "citations": [], "metadata": {"not_json": object()}},
    ],
)
def test_validate_mock_provider_response_rejects_invalid_values(value: object) -> None:
    with pytest.raises(InvalidMockProviderResponseError) as error:
        validate_mock_provider_response(value)

    assert error.value.code == "validation_error"


def test_dump_mock_provider_response_returns_json_payload() -> None:
    payload = dump_mock_provider_response(
        MockNormalResponse(
            answer="ok",
            citations=[MockCitation(source="docs", snippet="snippet")],
            metadata={"score": 1.0},
        )
    )

    assert payload == {
        "answer": "ok",
        "citations": [{"source": "docs", "snippet": "snippet"}],
        "metadata": {"score": 1.0},
    }
