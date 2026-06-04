from __future__ import annotations

import json
from pathlib import Path

from offline_llm_eval.util.document import DocumentLoadError, require_mapping


def load_json_document(text: str, *, source_name: str = "JSON") -> dict[str, object]:
    if text.strip() == "":
        raise DocumentLoadError(f"{source_name} document は空にできません。")

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DocumentLoadError(f"{source_name} document は不正なJSONです: {exc.msg}.") from exc

    return require_mapping(document, source_name)


def load_json_file(path: Path) -> dict[str, object]:
    return load_json_document(path.read_text(encoding="utf-8"), source_name=str(path))
