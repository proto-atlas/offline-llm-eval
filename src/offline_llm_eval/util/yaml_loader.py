from __future__ import annotations

from pathlib import Path

import yaml

from offline_llm_eval.util.document import DocumentLoadError, require_mapping


def load_yaml_document(text: str, *, source_name: str = "YAML") -> dict[str, object]:
    if text.strip() == "":
        raise DocumentLoadError(f"{source_name} document は空にできません。")

    try:
        documents = list(yaml.safe_load_all(text))
    except yaml.YAMLError as exc:
        raise DocumentLoadError(f"{source_name} document は不正なYAMLです。") from exc

    if len(documents) != 1:
        raise DocumentLoadError(f"{source_name} stream は1つのdocumentだけを含められます。")

    return require_mapping(documents[0], source_name)


def load_yaml_file(path: Path) -> dict[str, object]:
    return load_yaml_document(path.read_text(encoding="utf-8"), source_name=str(path))
