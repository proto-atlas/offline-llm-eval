import json
import logging
import sys
from io import StringIO
from uuid import UUID

import pytest

from offline_llm_eval.logging.structured import (
    InvalidRequestIdError,
    JsonLineFormatter,
    configure_structured_logging,
    resolve_request_id,
)

VALID_REQUEST_ID = "12345678-1234-5678-1234-567812345678"


def make_record() -> logging.LogRecord:
    record = logging.LogRecord(
        name="offline_llm_eval.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="providerが %s を返しました",
        args=("provider-secret-value",),
        exc_info=None,
    )
    record.created = 0.0
    record.request_id = VALID_REQUEST_ID
    record.structured = {
        "event": "provider_response",
        "metadata": {"api_key": "provider-secret-value"},
    }
    return record


def test_json_line_formatter_outputs_one_json_object_with_masked_values() -> None:
    formatted = JsonLineFormatter(secret_values=["provider-secret-value"]).format(make_record())

    assert json.loads(formatted) == {
        "event": "provider_response",
        "level": "INFO",
        "logger": "offline_llm_eval.test",
        "message": "providerが ******** を返しました",
        "metadata": {"api_key": "********"},
        "request_id": VALID_REQUEST_ID,
        "timestamp": "1970-01-01T00:00:00Z",
    }


def test_json_line_formatter_rejects_reserved_structured_field() -> None:
    record = make_record()
    record.structured = {"message": "reserved"}

    with pytest.raises(ValueError, match="構造化ログのfieldは予約済みです: message"):
        JsonLineFormatter().format(record)


def test_json_line_formatter_masks_exception_text() -> None:
    record = make_record()
    try:
        raise ValueError("provider-secret-value")
    except ValueError:
        record.exc_info = sys.exc_info()

    payload = json.loads(JsonLineFormatter(secret_values=["provider-secret-value"]).format(record))

    assert "provider-secret-value" not in payload["exception"]
    assert "ValueError: ********" in payload["exception"]


def test_json_line_formatter_rejects_invalid_request_id_field() -> None:
    record = make_record()
    record.request_id = "not-a-uuid"

    with pytest.raises(InvalidRequestIdError, match="X-Request-ID は有効なUUID"):
        JsonLineFormatter().format(record)


def test_configure_structured_logging_writes_to_provided_stream() -> None:
    stream = StringIO()
    logger = logging.getLogger("offline_llm_eval.stream_test")
    configure_structured_logging(logger=logger, stream=stream)

    logger.info("run started", extra={"request_id": VALID_REQUEST_ID})

    assert stream.getvalue().count("\n") == 1
    assert json.loads(stream.getvalue())["message"] == "run started"


def test_configure_structured_logging_masks_secret_values_in_stream() -> None:
    stream = StringIO()
    logger = logging.getLogger("offline_llm_eval.secret_stream_test")
    configure_structured_logging(
        logger=logger,
        stream=stream,
        secret_values=["provider-secret-value"],
    )

    logger.info(
        "providerが %s を返しました",
        "provider-secret-value",
        extra={
            "request_id": VALID_REQUEST_ID,
            "structured": {"metadata": ["provider-secret-value"]},
        },
    )

    output = stream.getvalue()
    payload = json.loads(output)

    assert output.count("\n") == 1
    assert "provider-secret-value" not in output
    assert payload["message"] == "providerが ******** を返しました"
    assert payload["metadata"] == ["********"]


def test_configure_structured_logging_uses_stdout_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    logger = logging.getLogger("offline_llm_eval.stdout_test")

    configure_structured_logging(logger=logger)
    logger.info("stdout message")

    assert json.loads(stream.getvalue())["message"] == "stdout message"


def test_resolve_request_id_adopts_valid_uuid() -> None:
    assert resolve_request_id("12345678123456781234567812345678") == VALID_REQUEST_ID


def test_resolve_request_id_generates_uuid_when_header_is_missing() -> None:
    assert resolve_request_id(None, new_id=lambda: UUID(VALID_REQUEST_ID)) == VALID_REQUEST_ID


def test_resolve_request_id_rejects_invalid_uuid() -> None:
    with pytest.raises(InvalidRequestIdError, match="X-Request-ID は有効なUUID"):
        resolve_request_id("not-a-uuid")
