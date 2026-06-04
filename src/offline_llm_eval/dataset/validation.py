from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class ResponseMode(StrEnum):
    JSON = "json"
    SSE = "sse"


MVP1_RESPONSE_MODE: ResponseMode = ResponseMode.JSON
RESPONSE_MODE_MISMATCH = "response_mode_mismatch"

type LocationPart = str | int


@dataclass(frozen=True, slots=True)
class CaseResponseContract:
    response_mode: ResponseMode
    expected_event_types: tuple[str, ...]
    skip_reason: str | None


class DatasetValidationError(ValueError):
    def __init__(self, message: str, loc: tuple[LocationPart, ...]) -> None:
        self.code = "validation_error"
        self.loc = loc
        super().__init__(f"{self.code}: {_format_loc(loc)}: {message}")


def validate_dataset_response_contracts(
    document: Mapping[str, object],
) -> list[CaseResponseContract]:
    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list):
        raise DatasetValidationError("cases はlistである必要があります。", ("cases",))

    contracts: list[CaseResponseContract] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise DatasetValidationError("case はmappingである必要があります。", ("cases", index))
        contracts.append(validate_case_response_contract(raw_case, case_index=index))

    return contracts


def validate_case_response_contract(
    case: Mapping[str, object],
    *,
    case_index: int | None = None,
) -> CaseResponseContract:
    response_mode = _parse_response_mode(
        case.get("expected_response_mode", ResponseMode.JSON.value), case_index
    )
    expected_event_types = _parse_expected_event_types(
        case.get("expected_event_types", []), case_index
    )

    if response_mode is ResponseMode.JSON and expected_event_types:
        raise DatasetValidationError(
            "expected_event_types は expected_response_mode=sse の場合のみ指定できます。",
            _case_loc(case_index, "expected_event_types"),
        )

    skip_reason = None
    if response_mode is not MVP1_RESPONSE_MODE:
        skip_reason = RESPONSE_MODE_MISMATCH

    return CaseResponseContract(
        response_mode=response_mode,
        expected_event_types=expected_event_types,
        skip_reason=skip_reason,
    )


def _parse_response_mode(value: object, case_index: int | None) -> ResponseMode:
    if not isinstance(value, str):
        raise DatasetValidationError(
            "expected_response_mode は文字列である必要があります。",
            _case_loc(case_index, "expected_response_mode"),
        )

    try:
        return ResponseMode(value)
    except ValueError as error:
        raise DatasetValidationError(
            "expected_response_mode は json または sse である必要があります。",
            _case_loc(case_index, "expected_response_mode"),
        ) from error


def _parse_expected_event_types(value: object, case_index: int | None) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DatasetValidationError(
            "expected_event_types はlistである必要があります。",
            _case_loc(case_index, "expected_event_types"),
        )

    event_types: list[str] = []
    for index, event_type in enumerate(value):
        if not isinstance(event_type, str):
            raise DatasetValidationError(
                "expected_event_types の要素は文字列である必要があります。",
                (*_case_loc(case_index, "expected_event_types"), index),
            )
        event_types.append(event_type)

    return tuple(event_types)


def _case_loc(case_index: int | None, field_name: str) -> tuple[LocationPart, ...]:
    if case_index is None:
        return (field_name,)
    return ("cases", case_index, field_name)


def _format_loc(loc: tuple[LocationPart, ...]) -> str:
    return ".".join(str(part) for part in loc)
