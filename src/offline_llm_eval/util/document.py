from __future__ import annotations

from collections.abc import Mapping
from typing import cast


class DocumentLoadError(ValueError):
    def __init__(self, message: str) -> None:
        self.code = "validation_error"
        super().__init__(f"{self.code}: {message}")


def require_mapping(document: object, source_name: str) -> dict[str, object]:
    if document is None:
        raise DocumentLoadError(f"{source_name} document は空にできません。")
    if not isinstance(document, Mapping):
        raise DocumentLoadError(f"{source_name} document はmappingである必要があります。")

    raw_mapping = cast(Mapping[object, object], document)
    result: dict[str, object] = {}
    for key, value in raw_mapping.items():
        if not isinstance(key, str):
            raise DocumentLoadError(f"{source_name} document のキーは文字列である必要があります。")
        result[key] = value

    return result
