from pathlib import Path

import pytest

from offline_llm_eval.dataset.validation import (
    RESPONSE_MODE_MISMATCH,
    DatasetValidationError,
    ResponseMode,
    validate_case_response_contract,
    validate_dataset_response_contracts,
)
from offline_llm_eval.util.yaml_loader import load_yaml_file


def test_validate_case_response_contract_accepts_json_without_events() -> None:
    contract = validate_case_response_contract({"expected_response_mode": "json"})

    assert contract.response_mode is ResponseMode.JSON
    assert contract.expected_event_types == ()
    assert contract.skip_reason is None


def test_validate_case_response_contract_accepts_json_with_empty_events() -> None:
    contract = validate_case_response_contract(
        {
            "expected_response_mode": "json",
            "expected_event_types": [],
        }
    )

    assert contract.response_mode is ResponseMode.JSON
    assert contract.expected_event_types == ()
    assert contract.skip_reason is None


def test_validate_case_response_contract_rejects_json_with_non_empty_events() -> None:
    with pytest.raises(DatasetValidationError) as error:
        validate_case_response_contract(
            {
                "expected_response_mode": "json",
                "expected_event_types": ["message.delta"],
            }
        )

    assert error.value.code == "validation_error"
    assert str(error.value) == (
        "validation_error: expected_event_types: "
        "expected_event_types は expected_response_mode=sse の場合のみ指定できます。"
    )


def test_validate_case_response_contract_marks_sse_as_mvp1_skip() -> None:
    contract = validate_case_response_contract(
        {
            "expected_response_mode": "sse",
            "expected_event_types": ["message.delta", "message.completed"],
        }
    )

    assert contract.response_mode is ResponseMode.SSE
    assert contract.expected_event_types == ("message.delta", "message.completed")
    assert contract.skip_reason == RESPONSE_MODE_MISMATCH


@pytest.mark.parametrize(
    ("field_value", "message"),
    [
        (1, "expected_response_mode は文字列である必要があります。"),
        ("xml", "expected_response_mode は json または sse である必要があります。"),
    ],
)
def test_validate_case_response_contract_rejects_invalid_response_mode(
    field_value: object,
    message: str,
) -> None:
    with pytest.raises(DatasetValidationError) as error:
        validate_case_response_contract({"expected_response_mode": field_value})

    assert str(error.value) == f"validation_error: expected_response_mode: {message}"


def test_validate_case_response_contract_rejects_non_list_event_types() -> None:
    with pytest.raises(DatasetValidationError) as error:
        validate_case_response_contract(
            {
                "expected_response_mode": "sse",
                "expected_event_types": "message.delta",
            }
        )

    assert str(error.value) == (
        "validation_error: expected_event_types: expected_event_types はlistである必要があります。"
    )


def test_validate_case_response_contract_rejects_non_string_event_type_item() -> None:
    with pytest.raises(DatasetValidationError) as error:
        validate_case_response_contract(
            {
                "expected_response_mode": "sse",
                "expected_event_types": ["message.delta", 1],
            },
            case_index=2,
        )

    assert str(error.value) == (
        "validation_error: cases.2.expected_event_types.1: "
        "expected_event_types の要素は文字列である必要があります。"
    )


def test_validate_dataset_response_contracts_validates_all_cases() -> None:
    contracts = validate_dataset_response_contracts(
        {
            "cases": [
                {"expected_response_mode": "json"},
                {
                    "expected_response_mode": "sse",
                    "expected_event_types": ["message.delta"],
                },
            ]
        }
    )

    assert [contract.response_mode for contract in contracts] == [
        ResponseMode.JSON,
        ResponseMode.SSE,
    ]
    assert [contract.skip_reason for contract in contracts] == [None, RESPONSE_MODE_MISMATCH]


def test_validate_dataset_response_contracts_rejects_non_list_cases() -> None:
    with pytest.raises(DatasetValidationError) as error:
        validate_dataset_response_contracts({"cases": {}})

    assert str(error.value) == "validation_error: cases: cases はlistである必要があります。"


def test_initial_dataset_response_contracts_are_valid_json_mode() -> None:
    dataset_paths = sorted(Path("datasets/initial").glob("*.yaml"))
    parsed = 0

    for path in dataset_paths:
        document = load_yaml_file(path)
        contracts = validate_dataset_response_contracts(document)
        parsed += len(contracts)
        assert {contract.response_mode for contract in contracts} == {ResponseMode.JSON}
        assert {contract.skip_reason for contract in contracts} == {None}

    assert parsed == 10
