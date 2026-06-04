import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from re import Pattern
from typing import Final, Protocol, runtime_checkable

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]

SECRET_SCAN_ASSERTION_ID: Final = "__secret_scan__"


class SecretPatternCategory(StrEnum):
    AWS_ACCESS_KEY = "aws_access_key"
    PEM_PRIVATE_KEY = "pem_private_key"
    JWT = "jwt"
    BEARER_TOKEN = "bearer_token"
    OPENAI_API_KEY = "openai_api_key"
    ANTHROPIC_API_KEY = "anthropic_api_key"
    GOOGLE_API_KEY = "google_api_key"
    GITHUB_PAT = "github_pat"
    SLACK_TOKEN = "slack_token"
    STRIPE_KEY = "stripe_key"
    GENERIC_PRIVATE_KEY = "generic_private_key"


class SecretScanStatus(StrEnum):
    PASS = "pass"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class SecretPattern:
    category: SecretPatternCategory
    pattern: Pattern[str]


@dataclass(frozen=True, slots=True)
class SecretScanResult:
    status: SecretScanStatus
    detail_code: str | None = None


@runtime_checkable
class ModelDumpable(Protocol):
    def model_dump(self, *, mode: str) -> object: ...


DEFAULT_SECRET_PATTERNS: Final = (
    SecretPattern(
        category=SecretPatternCategory.AWS_ACCESS_KEY,
        pattern=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    SecretPattern(
        category=SecretPatternCategory.PEM_PRIVATE_KEY,
        # パターン定義そのものが検出対象にならないよう、固定文字列を分けて保持する。
        pattern=re.compile("".join(("-----BEGIN ", "PRIVATE KEY-----"))),
    ),
    SecretPattern(
        category=SecretPatternCategory.JWT,
        pattern=re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    ),
    SecretPattern(
        category=SecretPatternCategory.BEARER_TOKEN,
        pattern=re.compile(r"\bBearer\s+ey[A-Za-z0-9._-]+"),
    ),
    SecretPattern(
        category=SecretPatternCategory.ANTHROPIC_API_KEY,
        pattern=re.compile(r"\bsk-ant-api03-[A-Za-z0-9_-]{20,}\b"),
    ),
    SecretPattern(
        category=SecretPatternCategory.GOOGLE_API_KEY,
        pattern=re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ),
    SecretPattern(
        category=SecretPatternCategory.GITHUB_PAT,
        pattern=re.compile(r"\b(?:ghp|gho)_[A-Za-z0-9_]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    ),
    SecretPattern(
        category=SecretPatternCategory.SLACK_TOKEN,
        pattern=re.compile(r"\bxox[bpars]-[A-Za-z0-9-]{10,}\b"),
    ),
    SecretPattern(
        category=SecretPatternCategory.STRIPE_KEY,
        pattern=re.compile(r"\b[sp]k_live_[A-Za-z0-9]{10,}\b"),
    ),
    SecretPattern(
        category=SecretPatternCategory.OPENAI_API_KEY,
        pattern=re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    SecretPattern(
        category=SecretPatternCategory.GENERIC_PRIVATE_KEY,
        # パターン定義そのものが検出対象にならないよう、固定文字列を分けて保持する。
        pattern=re.compile("".join((r"-----BEGIN [A-Z ]*", r"PRIVATE KEY-----"))),
    ),
)


def scan_secret_fields(value: object) -> SecretScanResult:
    return scan_secret_text(build_secret_scan_input(value))


def scan_secret_text(
    text: str,
    patterns: Sequence[SecretPattern] = DEFAULT_SECRET_PATTERNS,
) -> SecretScanResult:
    for secret_pattern in patterns:
        if secret_pattern.pattern.search(text) is not None:
            return SecretScanResult(
                status=SecretScanStatus.FAILED,
                detail_code=secret_pattern.category.value,
            )

    return SecretScanResult(status=SecretScanStatus.PASS)


def build_secret_scan_input(value: object) -> str:
    return "\n".join(collect_string_fields(value))


def collect_string_fields(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)

    if isinstance(value, ModelDumpable):
        return collect_string_fields(value.model_dump(mode="json"))

    if isinstance(value, Mapping):
        strings: list[str] = []
        for item in value.values():
            strings.extend(collect_string_fields(item))
        return tuple(strings)

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        strings = []
        for item in value:
            strings.extend(collect_string_fields(item))
        return tuple(strings)

    return ()
