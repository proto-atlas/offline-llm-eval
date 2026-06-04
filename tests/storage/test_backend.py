from pathlib import Path

import pytest

from offline_llm_eval.storage.backend import (
    InvalidStorageKeyError,
    LocalStorageBackend,
    StorageBackend,
    normalize_storage_key,
)


class InMemoryS3LikeBackend:
    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def write_bytes(self, key: str, data: bytes) -> None:
        self._objects[key] = data

    def read_bytes(self, key: str) -> bytes:
        return self._objects[key]

    def exists(self, key: str) -> bool:
        return key in self._objects


def test_local_storage_backend_writes_bytes_to_nested_key(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)

    backend.write_bytes("runs/1/evidence.md", b"evidence-body")

    assert (tmp_path / "runs" / "1" / "evidence.md").read_bytes() == b"evidence-body"


def test_local_storage_backend_reads_bytes_from_key(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    (tmp_path / "evidence.md").write_bytes(b"evidence-body")

    assert backend.read_bytes("evidence.md") == b"evidence-body"


def test_local_storage_backend_reports_existing_file(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    (tmp_path / "evidence.md").write_bytes(b"evidence-body")

    assert backend.exists("evidence.md") is True


def test_local_storage_backend_reports_missing_file(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)

    assert backend.exists("missing.md") is False


def test_normalize_storage_key_converts_windows_separator() -> None:
    assert normalize_storage_key("runs\\1\\evidence.md") == "runs/1/evidence.md"


@pytest.mark.parametrize("key", ["", " ", "/absolute.md", "../escape.md", "runs/../escape.md"])
def test_normalize_storage_key_rejects_unsafe_key(key: str) -> None:
    with pytest.raises(InvalidStorageKeyError):
        normalize_storage_key(key)


def test_local_storage_backend_matches_storage_backend_protocol(tmp_path: Path) -> None:
    backend: StorageBackend = LocalStorageBackend(tmp_path)

    assert_storage_backend_contract(backend)


def test_s3_shaped_backend_matches_storage_backend_protocol() -> None:
    backend: StorageBackend = InMemoryS3LikeBackend()

    assert_storage_backend_contract(backend)


def assert_storage_backend_contract(backend: StorageBackend) -> None:
    backend.write_bytes("runs/1/evidence.md", b"evidence-body")

    assert backend.exists("runs/1/evidence.md") is True
    assert backend.read_bytes("runs/1/evidence.md") == b"evidence-body"
    assert backend.exists("runs/1/missing.md") is False
