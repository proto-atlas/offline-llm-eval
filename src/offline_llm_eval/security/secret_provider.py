from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

REDACTED_SECRET = "********"


class SecretNotFoundError(LookupError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Secret が設定されていません: {name}")


@dataclass(frozen=True, repr=False)
class SecretValue:
    name: str
    _value: str

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"SecretValue(name={self.name!r}, value={REDACTED_SECRET!r})"

    def __str__(self) -> str:
        return REDACTED_SECRET


class EnvSecretProvider:
    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self._env = os.environ if env is None else env

    def get(self, name: str) -> SecretValue:
        normalized_name = name.strip()
        if normalized_name == "":
            raise ValueError("Secret name は空にできません。")

        value = self._env.get(normalized_name)
        if value is None or value == "":
            raise SecretNotFoundError(normalized_name)

        return SecretValue(name=normalized_name, _value=value)
