from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, cast

from pydantic import ValidationError
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
    text,
    true,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from offline_llm_eval.dataset.assertion_model import (
    AssertionRecord,
    AssertionSeverity,
    AssertionYaml,
    resolve_assertion_severity,
)
from offline_llm_eval.dataset.repository import (
    DatasetRecord,
    DatasetRepository,
    JsonObject,
    JsonValue,
)
from offline_llm_eval.dataset.validation import (
    DatasetValidationError,
    validate_case_response_contract,
)
from offline_llm_eval.dataset.version import InvalidDatasetVersionError, parse_dataset_version
from offline_llm_eval.db.base import Base
from offline_llm_eval.provider.mock_behavior import (
    MISSING_MOCK_RESPONSE,
    InvalidMockResponseError,
    parse_mock_response,
)
from offline_llm_eval.util.json_loader import load_json_file
from offline_llm_eval.util.yaml_loader import load_yaml_file

YAML_SUFFIXES: Final = {".yaml", ".yml"}
JSON_FILE_SUFFIX: Final = ".json"

type LocationPart = str | int


class EvaluationCaseRecord(Base):
    __tablename__ = "evaluation_cases"
    __table_args__ = (
        UniqueConstraint("dataset_id", "case_key", name="uq_evaluation_cases_dataset_case_key"),
        CheckConstraint(
            "severity in ('low', 'medium', 'high')",
            name="ck_evaluation_cases_severity",
        ),
    )

    case_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.dataset_id", ondelete="RESTRICT"),
        nullable=False,
    )
    case_key: Mapped[str] = mapped_column(String(255), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    tags_json: Mapped[list[str]] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[JsonObject | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=true(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


@dataclass(frozen=True, slots=True)
class DatasetCaseInput:
    case_key: str
    question: str
    severity: AssertionSeverity
    tags_json: list[str]
    metadata_json: JsonObject | None
    is_active: bool
    assertions: tuple[AssertionYaml, ...]


@dataclass(frozen=True, slots=True)
class DatasetImportInput:
    name: str
    dataset_version: str
    metadata_json: JsonObject | None
    cases: tuple[DatasetCaseInput, ...]


@dataclass(frozen=True, slots=True)
class DatasetImportResult:
    dataset_id: int
    name: str
    dataset_version: str
    case_count: int
    assertion_count: int


class DatasetImportError(ValueError):
    def __init__(self, message: str, loc: tuple[LocationPart, ...]) -> None:
        self.code = "validation_error"
        self.loc = loc
        super().__init__(f"{self.code}: {_format_loc(loc)}: {message}")


async def import_dataset_file(session: AsyncSession, path: Path) -> DatasetImportResult:
    document = _load_dataset_file(path)
    return await import_dataset_document(session, document)


async def import_dataset_document(
    session: AsyncSession,
    document: Mapping[str, object],
) -> DatasetImportResult:
    dataset_input = parse_dataset_import_document(document)
    dataset = await DatasetRepository(session).get_or_create_dataset(
        dataset_input.name,
        dataset_input.dataset_version,
        metadata_json=dataset_input.metadata_json,
    )

    assertion_count = 0
    for case_input in dataset_input.cases:
        case_record = await _upsert_case(session, dataset.dataset_id, case_input)
        for assertion in case_input.assertions:
            await _upsert_assertion(session, case_record.case_id, case_input.severity, assertion)
            assertion_count += 1

    return _to_result(dataset, len(dataset_input.cases), assertion_count)


def parse_dataset_import_document(document: Mapping[str, object]) -> DatasetImportInput:
    name = _required_str(document.get("name"), ("name",))
    dataset_version = _parse_dataset_version(document.get("dataset_version"))
    metadata_json = _optional_json_object(document.get("metadata"), ("metadata",))
    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list):
        raise DatasetImportError("cases はlistである必要があります。", ("cases",))

    seen_case_keys: set[str] = set()
    cases: list[DatasetCaseInput] = []
    for index, raw_case in enumerate(raw_cases):
        case = _require_mapping(raw_case, ("cases", index))
        case_input = _parse_case(case, index)
        if case_input.case_key in seen_case_keys:
            raise DatasetImportError(
                "case_key はデータセット内で一意である必要があります。",
                ("cases", index, "case_key"),
            )
        seen_case_keys.add(case_input.case_key)
        cases.append(case_input)

    return DatasetImportInput(
        name=name,
        dataset_version=dataset_version,
        metadata_json=metadata_json,
        cases=tuple(cases),
    )


def _load_dataset_file(path: Path) -> dict[str, object]:
    suffix = path.suffix.lower()
    if suffix in YAML_SUFFIXES:
        return load_yaml_file(path)
    if suffix == JSON_FILE_SUFFIX:
        return load_json_file(path)

    raise DatasetImportError(
        "dataset file は .yaml, .yml, .json のいずれかである必要があります。",
        ("path",),
    )


def _parse_dataset_version(value: object) -> str:
    try:
        return parse_dataset_version(value).value
    except InvalidDatasetVersionError as error:
        raise DatasetImportError(str(error), ("dataset_version",)) from error


def _parse_case(case: Mapping[str, object], case_index: int) -> DatasetCaseInput:
    _validate_response_contract(case, case_index)
    expected_answer = _optional_str(
        case.get("expected_answer"), ("cases", case_index, "expected_answer")
    )
    _validate_mock_response(case, case_index, expected_answer)

    case_severity = _parse_case_severity(
        case.get("severity", AssertionSeverity.MEDIUM.value), case_index
    )
    return DatasetCaseInput(
        case_key=_required_str(case.get("case_key"), ("cases", case_index, "case_key")),
        question=_required_str(case.get("question"), ("cases", case_index, "question")),
        severity=case_severity,
        tags_json=_parse_tags(case.get("tags", []), case_index),
        metadata_json=_optional_json_object(
            case.get("metadata"), ("cases", case_index, "metadata")
        ),
        is_active=_parse_bool(case.get("is_active", True), ("cases", case_index, "is_active")),
        assertions=_parse_assertions(case.get("assertions", []), case_index),
    )


def _validate_response_contract(case: Mapping[str, object], case_index: int) -> None:
    try:
        validate_case_response_contract(case, case_index=case_index)
    except DatasetValidationError as error:
        raise DatasetImportError("応答契約が不正です。", error.loc) from error


def _validate_mock_response(
    case: Mapping[str, object],
    case_index: int,
    expected_answer: str | None,
) -> None:
    mock_response = case["mock_response"] if "mock_response" in case else MISSING_MOCK_RESPONSE
    try:
        parse_mock_response(mock_response, expected_answer=expected_answer)
    except InvalidMockResponseError as error:
        raise DatasetImportError(
            "mock_response が不正です。",
            ("cases", case_index, "mock_response"),
        ) from error


def _parse_case_severity(value: object, case_index: int) -> AssertionSeverity:
    if not isinstance(value, str):
        raise DatasetImportError(
            "severity は文字列である必要があります。",
            ("cases", case_index, "severity"),
        )

    try:
        return AssertionSeverity(value)
    except ValueError as error:
        raise DatasetImportError(
            "severity は low, medium, high のいずれかである必要があります。",
            ("cases", case_index, "severity"),
        ) from error


def _parse_tags(value: object, case_index: int) -> list[str]:
    if not isinstance(value, list):
        raise DatasetImportError("tags はlistである必要があります。", ("cases", case_index, "tags"))

    tags: list[str] = []
    for index, tag in enumerate(value):
        if not isinstance(tag, str):
            raise DatasetImportError(
                "tags の要素は文字列である必要があります。",
                ("cases", case_index, "tags", index),
            )
        tags.append(tag)

    return tags


def _parse_assertions(value: object, case_index: int) -> tuple[AssertionYaml, ...]:
    if not isinstance(value, list):
        raise DatasetImportError(
            "assertions はlistである必要があります。",
            ("cases", case_index, "assertions"),
        )

    seen_assertion_ids: set[str] = set()
    assertions: list[AssertionYaml] = []
    for index, raw_assertion in enumerate(value):
        assertion_mapping = _require_mapping(
            raw_assertion, ("cases", case_index, "assertions", index)
        )
        try:
            assertion = AssertionYaml.model_validate(assertion_mapping)
        except ValidationError as error:
            raise DatasetImportError(
                "assertion が不正です。",
                ("cases", case_index, "assertions", index),
            ) from error

        if assertion.id in seen_assertion_ids:
            raise DatasetImportError(
                "assertion id はケース内で一意である必要があります。",
                ("cases", case_index, "assertions", index, "id"),
            )
        seen_assertion_ids.add(assertion.id)
        assertions.append(assertion)

    return tuple(assertions)


def _required_str(value: object, loc: tuple[LocationPart, ...]) -> str:
    if not isinstance(value, str):
        raise DatasetImportError("文字列である必要があります。", loc)
    if value == "":
        raise DatasetImportError("空にできません。", loc)
    if value.strip() != value:
        raise DatasetImportError("前後に空白を含められません。", loc)

    return value


def _optional_str(value: object, loc: tuple[LocationPart, ...]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetImportError("文字列またはnullである必要があります。", loc)

    return value


def _parse_bool(value: object, loc: tuple[LocationPart, ...]) -> bool:
    if not isinstance(value, bool):
        raise DatasetImportError("booleanである必要があります。", loc)

    return value


def _optional_json_object(value: object, loc: tuple[LocationPart, ...]) -> JsonObject | None:
    if value is None:
        return None

    mapping = _require_mapping(value, loc)
    json_object: JsonObject = {}
    for key, raw_value in mapping.items():
        json_object[key] = _json_value(raw_value, (*loc, key))

    return json_object


def _json_value(value: object, loc: tuple[LocationPart, ...]) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, list):
        return [_json_value(item, (*loc, index)) for index, item in enumerate(value)]

    if isinstance(value, Mapping):
        raw_mapping = cast(Mapping[object, object], value)
        json_object: JsonObject = {}
        for key, raw_value in raw_mapping.items():
            if not isinstance(key, str):
                raise DatasetImportError("objectのキーは文字列である必要があります。", loc)
            json_object[key] = _json_value(raw_value, (*loc, key))
        return json_object

    raise DatasetImportError("JSON互換値である必要があります。", loc)


def _require_mapping(value: object, loc: tuple[LocationPart, ...]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DatasetImportError("mappingである必要があります。", loc)

    raw_mapping = cast(Mapping[object, object], value)
    result: dict[str, object] = {}
    for key, raw_value in raw_mapping.items():
        if not isinstance(key, str):
            raise DatasetImportError("mappingのキーは文字列である必要があります。", loc)
        result[key] = raw_value

    return result


async def _upsert_case(
    session: AsyncSession,
    dataset_id: int,
    case_input: DatasetCaseInput,
) -> EvaluationCaseRecord:
    result = await session.execute(
        select(EvaluationCaseRecord).where(
            EvaluationCaseRecord.dataset_id == dataset_id,
            EvaluationCaseRecord.case_key == case_input.case_key,
        )
    )
    case_record = result.scalar_one_or_none()
    if case_record is None:
        case_record = EvaluationCaseRecord(
            dataset_id=dataset_id,
            case_key=case_input.case_key,
            question=case_input.question,
            severity=case_input.severity.value,
            tags_json=case_input.tags_json,
            metadata_json=case_input.metadata_json,
            is_active=case_input.is_active,
        )
        session.add(case_record)
    else:
        case_record.question = case_input.question
        case_record.severity = case_input.severity.value
        case_record.tags_json = case_input.tags_json
        case_record.metadata_json = case_input.metadata_json
        case_record.is_active = case_input.is_active

    await session.flush()
    return case_record


async def _upsert_assertion(
    session: AsyncSession,
    case_id: int,
    case_severity: AssertionSeverity,
    assertion: AssertionYaml,
) -> AssertionRecord:
    result = await session.execute(
        select(AssertionRecord).where(
            AssertionRecord.case_id == case_id,
            AssertionRecord.id == assertion.id,
        )
    )
    assertion_record = result.scalar_one_or_none()
    severity = resolve_assertion_severity(assertion, case_severity)
    if assertion_record is None:
        assertion_record = AssertionRecord(
            case_id=case_id,
            id=assertion.id,
            assertion_type=assertion.assertion_type.value,
            expected_json=assertion.expected,
            required=assertion.required,
            on_fail=assertion.on_fail.value,
            severity=severity.value,
            is_active=assertion.is_active,
        )
        session.add(assertion_record)
    else:
        assertion_record.assertion_type = assertion.assertion_type.value
        assertion_record.expected_json = assertion.expected
        assertion_record.required = assertion.required
        assertion_record.on_fail = assertion.on_fail.value
        assertion_record.severity = severity.value
        assertion_record.is_active = assertion.is_active

    await session.flush()
    return assertion_record


def _to_result(
    dataset: DatasetRecord,
    case_count: int,
    assertion_count: int,
) -> DatasetImportResult:
    return DatasetImportResult(
        dataset_id=dataset.dataset_id,
        name=dataset.name,
        dataset_version=dataset.dataset_version,
        case_count=case_count,
        assertion_count=assertion_count,
    )


def _format_loc(loc: tuple[LocationPart, ...]) -> str:
    return ".".join(str(part) for part in loc)
