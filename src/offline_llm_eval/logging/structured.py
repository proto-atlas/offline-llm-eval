from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import TextIO, cast
from uuid import UUID, uuid4

from offline_llm_eval.security.secret_provider import REDACTED_SECRET

REQUEST_ID_FIELD = "request_id"
STRUCTURED_FIELDS = "structured"

BASE_LOG_FIELDS = frozenset({"timestamp", "level", "logger", "message", REQUEST_ID_FIELD})


class InvalidRequestIdError(ValueError):
    def __init__(self) -> None:
        super().__init__("X-Request-ID は有効なUUIDである必要があります。")


def resolve_request_id(
    header_value: str | None,
    *,
    new_id: Callable[[], UUID] = uuid4,
) -> str:
    if header_value is None or header_value.strip() == "":
        return str(new_id())

    try:
        return str(UUID(header_value))
    except ValueError as exc:
        raise InvalidRequestIdError() from exc


def mask_secret_values(value: object, secret_values: Iterable[str]) -> object:
    secrets = tuple(secret_value for secret_value in secret_values if secret_value != "")
    return _mask_value(value, secrets)


def _mask_value(value: object, secrets: tuple[str, ...]) -> object:
    if isinstance(value, str):
        masked = value
        for secret in secrets:
            masked = masked.replace(secret, REDACTED_SECRET)
        return masked

    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _mask_value(item, secrets) for key, item in mapping.items()}

    if isinstance(value, list | tuple):
        return [_mask_value(item, secrets) for item in value]

    return value


class JsonLineFormatter(logging.Formatter):
    def __init__(self, *, secret_values: Iterable[str] = ()) -> None:
        super().__init__()
        self._secret_values = tuple(
            secret_value for secret_value in secret_values if secret_value != ""
        )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": _format_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": mask_secret_values(record.getMessage(), self._secret_values),
        }

        request_id = getattr(record, REQUEST_ID_FIELD, None)
        if isinstance(request_id, str) and request_id != "":
            payload[REQUEST_ID_FIELD] = resolve_request_id(request_id)

        for key, value in _get_structured_fields(record).items():
            if key in BASE_LOG_FIELDS:
                raise ValueError(f"構造化ログのfieldは予約済みです: {key}")
            payload[key] = mask_secret_values(value, self._secret_values)

        if record.exc_info is not None:
            payload["exception"] = mask_secret_values(
                self.formatException(record.exc_info),
                self._secret_values,
            )

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def configure_structured_logging(
    *,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    stream: TextIO | None = None,
    secret_values: Iterable[str] = (),
) -> logging.Logger:
    target_logger = logging.getLogger() if logger is None else logger
    output_stream = sys.stdout if stream is None else stream
    handler = logging.StreamHandler(output_stream)
    handler.setFormatter(JsonLineFormatter(secret_values=secret_values))

    target_logger.handlers.clear()
    target_logger.addHandler(handler)
    target_logger.setLevel(level)
    target_logger.propagate = False
    return target_logger


def _format_timestamp(created: float) -> str:
    return datetime.fromtimestamp(created, UTC).isoformat().replace("+00:00", "Z")


def _get_structured_fields(record: logging.LogRecord) -> dict[str, object]:
    raw_fields = getattr(record, STRUCTURED_FIELDS, None)
    if raw_fields is None:
        return {}
    if not isinstance(raw_fields, Mapping):
        raise TypeError("構造化ログのfieldsはmappingである必要があります。")

    fields = cast(Mapping[object, object], raw_fields)
    return {str(key): value for key, value in fields.items()}
