import json

import pytest

from offline_llm_eval.api.error_response import (
    ERROR_DEFINITIONS,
    ApiErrorCode,
    InvalidErrorExtraFieldsError,
    build_error_body,
    build_error_response,
)


def test_error_definitions_match_nfr_010_table() -> None:
    assert ERROR_DEFINITIONS == {
        ApiErrorCode.BASELINE_NOT_FOUND: (404, frozenset({"baseline_spec"})),
        ApiErrorCode.BASELINE_IN_PROGRESS: (409, frozenset({"run_id"})),
        ApiErrorCode.CONCURRENT_RUN_BLOCKED: (409, frozenset({"dataset_id", "target_label"})),
        ApiErrorCode.DATASET_IMPORT_BLOCKED_BY_RUNNING_RUN: (
            409,
            frozenset({"dataset_id", "running_run_id"}),
        ),
        ApiErrorCode.LOCK_TIMEOUT: (503, frozenset()),
        ApiErrorCode.VALIDATION_ERROR: (400, frozenset({"errors"})),
        ApiErrorCode.CASE_NOT_FOUND: (404, frozenset({"run_id", "case_key"})),
        ApiErrorCode.RUN_NOT_FOUND: (404, frozenset({"run_id"})),
    }


def test_build_error_body_returns_common_schema() -> None:
    body = build_error_body(
        ApiErrorCode.BASELINE_NOT_FOUND,
        "基準実行が見つかりません。",
        {"baseline_spec": "latest_main"},
    )

    assert body == {
        "error": {
            "code": "baseline_not_found",
            "message": "基準実行が見つかりません。",
            "extra": {"baseline_spec": "latest_main"},
        },
    }


def test_build_error_body_rejects_missing_extra_fields() -> None:
    with pytest.raises(InvalidErrorExtraFieldsError, match="missing=\\['baseline_spec'\\]"):
        build_error_body(ApiErrorCode.BASELINE_NOT_FOUND, "基準実行が見つかりません。")


def test_build_error_body_rejects_unexpected_extra_fields() -> None:
    with pytest.raises(InvalidErrorExtraFieldsError, match="unexpected=\\['run_id'\\]"):
        build_error_body(
            ApiErrorCode.LOCK_TIMEOUT,
            "database lock timeout",
            {"run_id": "run-1"},
        )


def test_build_error_response_uses_status_code_and_body() -> None:
    response = build_error_response(
        ApiErrorCode.RUN_NOT_FOUND,
        "実行が見つかりません。",
        {"run_id": "run-1"},
    )

    assert response.status_code == 404
    assert json.loads(bytes(response.body)) == {
        "error": {
            "code": "run_not_found",
            "message": "実行が見つかりません。",
            "extra": {"run_id": "run-1"},
        },
    }
