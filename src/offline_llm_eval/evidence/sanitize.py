from collections.abc import Mapping
from typing import Final

from offline_llm_eval.dataset.repository import JsonValue
from offline_llm_eval.evaluator.secret_pattern import (
    DEFAULT_SECRET_PATTERNS,
    SecretPattern,
)

MASKED_SECRET_PREFIX: Final = "[masked:"
MASKED_SECRET_SUFFIX: Final = "]"
REVIEWER_VERDICT_MAX_LENGTH: Final = 64
REVIEWER_VERDICT_LENGTH_MASK: Final = "[masked:reviewer_verdict]"


def sanitize_evidence_text(
    text: str,
    patterns: tuple[SecretPattern, ...] = DEFAULT_SECRET_PATTERNS,
) -> str:
    sanitized = text
    for secret_pattern in patterns:
        sanitized = secret_pattern.pattern.sub(
            _masked_secret(secret_pattern.category.value),
            sanitized,
        )
    return sanitized


def sanitize_reviewer_note(note: str | None) -> str | None:
    if note is None:
        return None
    return sanitize_evidence_text(note)


def sanitize_reviewer_verdict(reviewer_verdict: str) -> str:
    sanitized = sanitize_evidence_text(reviewer_verdict)
    if len(sanitized) > REVIEWER_VERDICT_MAX_LENGTH:
        return REVIEWER_VERDICT_LENGTH_MASK
    return sanitized


def sanitize_optional_reviewer_verdict(reviewer_verdict: str | None) -> str | None:
    if reviewer_verdict is None:
        return None
    return sanitize_reviewer_verdict(reviewer_verdict)


def sanitize_evidence_value(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return sanitize_evidence_text(value)

    if isinstance(value, list):
        return [sanitize_evidence_value(item) for item in value]

    if isinstance(value, Mapping):
        return {
            sanitize_evidence_text(str(key)): sanitize_evidence_value(item)
            for key, item in value.items()
        }

    return value


def _masked_secret(category: str) -> str:
    return f"{MASKED_SECRET_PREFIX}{category}{MASKED_SECRET_SUFFIX}"
