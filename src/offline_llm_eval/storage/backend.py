from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Protocol


class InvalidStorageKeyError(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(f"Storage key は相対パスである必要があります: {key}")


class StorageBackend(Protocol):
    def write_bytes(self, key: str, data: bytes) -> None: ...

    def read_bytes(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


class LocalStorageBackend:
    def __init__(self, root: Path) -> None:
        self._root = root

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read_bytes(self, key: str) -> bytes:
        return self._path_for(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path_for(key).is_file()

    def _path_for(self, key: str) -> Path:
        storage_key = normalize_storage_key(key)
        return self._root.joinpath(*PurePosixPath(storage_key).parts)


def normalize_storage_key(key: str) -> str:
    normalized = key.strip().replace("\\", "/")
    path = PurePosixPath(normalized)

    if normalized == "" or path.is_absolute() or ".." in path.parts:
        raise InvalidStorageKeyError(key)

    return path.as_posix()
