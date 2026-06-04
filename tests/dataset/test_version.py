import pytest

from offline_llm_eval.dataset.version import (
    InvalidDatasetVersionError,
    is_semver_version,
    parse_dataset_version,
)


def test_parse_dataset_version_marks_semver_without_changing_value() -> None:
    version = parse_dataset_version("1.2.3-alpha.1+build.5")

    assert version.value == "1.2.3-alpha.1+build.5"
    assert version.is_semver is True


@pytest.mark.parametrize(
    "value",
    [
        "release candidate 1",
        "01.0.0",
        "v1.2.3",
    ],
)
def test_parse_dataset_version_allows_arbitrary_non_blank_string(value: str) -> None:
    version = parse_dataset_version(value)

    assert version.value == value
    assert version.is_semver is False


def test_parse_dataset_version_preserves_exact_unique_key_value() -> None:
    lower = parse_dataset_version("1.0.0+build.1")
    upper = parse_dataset_version("1.0.0+BUILD.1")

    assert lower.value == "1.0.0+build.1"
    assert upper.value == "1.0.0+BUILD.1"
    assert lower.value != upper.value
    assert lower.is_semver is True
    assert upper.is_semver is True


def test_is_semver_version_uses_semver_full_string_match() -> None:
    assert is_semver_version("1.0.0") is True
    assert is_semver_version("1.0.0 trailing") is False


def test_parse_dataset_version_rejects_empty_string() -> None:
    with pytest.raises(InvalidDatasetVersionError) as error:
        parse_dataset_version("")

    assert error.value.code == "validation_error"
    assert str(error.value) == "validation_error: dataset_version は空にできません。"


def test_parse_dataset_version_rejects_leading_or_trailing_whitespace() -> None:
    with pytest.raises(InvalidDatasetVersionError) as error:
        parse_dataset_version(" 1.0.0")

    assert str(error.value) == ("validation_error: dataset_version は前後に空白を含められません。")


def test_parse_dataset_version_rejects_non_string_value() -> None:
    with pytest.raises(InvalidDatasetVersionError) as error:
        parse_dataset_version(1)

    assert str(error.value) == "validation_error: dataset_version は文字列である必要があります。"
