from collections.abc import Callable
from pathlib import Path

import pytest

from offline_llm_eval.util.document import DocumentLoadError
from offline_llm_eval.util.json_loader import load_json_document, load_json_file
from offline_llm_eval.util.yaml_loader import load_yaml_document, load_yaml_file


def test_load_json_document_returns_mapping() -> None:
    assert load_json_document('{"name": "dataset", "version": 1}') == {
        "name": "dataset",
        "version": 1,
    }


def test_load_yaml_document_returns_mapping() -> None:
    assert load_yaml_document("name: dataset\nversion: 1\n") == {
        "name": "dataset",
        "version": 1,
    }


DocumentLoader = Callable[[str], dict[str, object]]


@pytest.mark.parametrize(
    ("loader", "label"),
    [
        (load_json_document, "JSON"),
        (load_yaml_document, "YAML"),
    ],
)
def test_load_document_rejects_empty_text(loader: DocumentLoader, label: str) -> None:

    with pytest.raises(DocumentLoadError) as error:
        loader(" ")

    assert error.value.code == "validation_error"
    assert str(error.value) == f"validation_error: {label} document は空にできません。"


def test_load_json_document_rejects_non_mapping() -> None:
    with pytest.raises(DocumentLoadError) as error:
        load_json_document("[]")

    assert str(error.value) == "validation_error: JSON document はmappingである必要があります。"


def test_load_yaml_document_rejects_non_mapping() -> None:
    with pytest.raises(DocumentLoadError) as error:
        load_yaml_document("- item\n")

    assert str(error.value) == "validation_error: YAML document はmappingである必要があります。"


def test_load_yaml_document_rejects_multiple_documents() -> None:
    with pytest.raises(DocumentLoadError) as error:
        load_yaml_document("name: first\n---\nname: second\n")

    assert str(error.value) == "validation_error: YAML stream は1つのdocumentだけを含められます。"


def test_load_yaml_document_rejects_unsafe_tag() -> None:
    with pytest.raises(DocumentLoadError) as error:
        load_yaml_document("!!python/object/apply:os.system ['echo unsafe']")

    assert str(error.value) == "validation_error: YAML document は不正なYAMLです。"


def test_load_json_document_rejects_invalid_json() -> None:
    with pytest.raises(DocumentLoadError) as error:
        load_json_document("{")

    assert (
        str(error.value) == "validation_error: JSON document は不正なJSONです: "
        "Expecting property name enclosed in double quotes."
    )


def test_load_json_file_reads_utf8_text(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text('{"name": "dataset"}', encoding="utf-8")

    assert load_json_file(path) == {"name": "dataset"}


def test_load_yaml_file_reads_utf8_text(tmp_path: Path) -> None:
    path = tmp_path / "dataset.yaml"
    path.write_text("name: dataset\n", encoding="utf-8")

    assert load_yaml_file(path) == {"name": "dataset"}
