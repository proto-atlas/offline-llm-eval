from pathlib import Path

import pytest

from offline_llm_eval.provider.mock_behavior import (
    InvalidMockResponseError,
    MockErrorCode,
    MockErrorEnvelope,
    MockErrorResponse,
    MockNormalResponse,
    MockResponseSource,
    parse_mock_response,
    select_mock_response_variant,
)
from offline_llm_eval.util.yaml_loader import load_yaml_file


def test_parse_mock_response_uses_default_when_field_is_omitted() -> None:
    behavior = parse_mock_response(expected_answer="expected answer")

    assert behavior.source is MockResponseSource.DEFAULT
    assert isinstance(behavior.response, MockNormalResponse)
    assert behavior.response.answer == "expected answer"
    assert behavior.response.citations == []
    assert behavior.response.metadata == {}


def test_parse_mock_response_uses_empty_default_for_missing_expected_answer() -> None:
    behavior = parse_mock_response(expected_answer=None)

    assert isinstance(behavior.response, MockNormalResponse)
    assert behavior.response.answer == ""


def test_parse_mock_response_accepts_normal_response_object() -> None:
    behavior = parse_mock_response(
        {
            "answer": "The health endpoints are ready.",
            "citations": [
                {
                    "source": "docs/health",
                    "snippet": "GET /health returns liveness.",
                }
            ],
            "metadata": {"fixture": True},
        }
    )

    assert behavior.source is MockResponseSource.EXPLICIT
    assert isinstance(behavior.response, MockNormalResponse)
    assert behavior.response.answer == "The health endpoints are ready."
    assert behavior.response.citations[0].source == "docs/health"
    assert behavior.response.metadata == {"fixture": True}


def test_parse_mock_response_accepts_error_envelope_object() -> None:
    behavior = parse_mock_response(
        {
            "error": {
                "code": "provider_error",
                "message": "providerが制御されたエラーを返しました。",
            }
        }
    )

    assert behavior.source is MockResponseSource.EXPLICIT
    assert isinstance(behavior.response, MockErrorResponse)
    assert behavior.response.error.code is MockErrorCode.PROVIDER_ERROR
    assert behavior.response.error.message == "providerが制御されたエラーを返しました。"


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not-an-object",
        [],
        {},
        {"answer": "missing citations"},
        {"answer": "x", "citations": [], "unknown": True},
        {"answer": "x", "citations": [], "events": [{"type": "delta"}]},
        {"error": {}},
        {"error": {"message": "x"}},
        {"error": {"code": 123, "message": "x"}},
        {"error": {"code": "foo", "message": "x"}},
        {"answer": "x", "citations": [], "error": "scalar"},
    ],
)
def test_parse_mock_response_rejects_invalid_values(value: object) -> None:
    with pytest.raises(InvalidMockResponseError) as error:
        parse_mock_response(value)

    assert error.value.code == "validation_error"


def test_callable_discriminator_handles_dict_and_model_instances() -> None:
    normal = MockNormalResponse(answer="ok", citations=[], metadata={})
    error = MockErrorResponse(
        error=MockErrorEnvelope(
            code=MockErrorCode.OVERLOADED,
            message="providerが過負荷を返しました。",
        )
    )

    assert select_mock_response_variant({"answer": "ok", "citations": []}) == "normal"
    assert (
        select_mock_response_variant({"error": {"code": "overloaded", "message": "x"}}) == "error"
    )
    assert select_mock_response_variant(normal) == "normal"
    assert select_mock_response_variant(error) == "error"


def test_initial_dataset_mock_responses_match_schema() -> None:
    dataset_paths = sorted(Path("datasets/initial").glob("*.yaml"))
    parsed = 0

    for path in dataset_paths:
        document = load_yaml_file(path)
        cases = document["cases"]
        assert isinstance(cases, list)
        for case in cases:
            assert isinstance(case, dict)
            behavior = parse_mock_response(case.get("mock_response"))
            assert behavior.source is MockResponseSource.EXPLICIT
            parsed += 1

    assert parsed == 10
