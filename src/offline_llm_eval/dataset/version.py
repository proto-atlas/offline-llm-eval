import re
from dataclasses import dataclass
from typing import Final

SEMVER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    value: str
    is_semver: bool


class InvalidDatasetVersionError(ValueError):
    code = "validation_error"

    def __init__(self, message: str) -> None:
        super().__init__(f"{self.code}: {message}")


def is_semver_version(value: str) -> bool:
    return SEMVER_PATTERN.fullmatch(value) is not None


def parse_dataset_version(value: object) -> DatasetVersion:
    if not isinstance(value, str):
        raise InvalidDatasetVersionError("dataset_version は文字列である必要があります。")

    if value == "":
        raise InvalidDatasetVersionError("dataset_version は空にできません。")

    if value.strip() != value:
        raise InvalidDatasetVersionError("dataset_version は前後に空白を含められません。")

    return DatasetVersion(value=value, is_semver=is_semver_version(value))
