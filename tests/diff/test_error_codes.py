import pytest

from offline_llm_eval.diff.error_codes import (
    compare_error_type_counts,
    count_error_types,
    empty_error_type_counts,
)
from offline_llm_eval.runner.error_type import ErrorType


def test_error_type別件数を集計する() -> None:
    counts = count_error_types(
        (
            ErrorType.PROVIDER_ERROR,
            "provider_error",
            ErrorType.OVERLOADED,
            None,
            "response_mode_mismatch",
        )
    )

    assert counts == {
        "provider_error": 2,
        "overloaded": 1,
        "unknown_error": 0,
        "response_mode_mismatch": 1,
    }


def test_baselineとcurrentのerror_type差分を返す() -> None:
    comparison = compare_error_type_counts(
        baseline_error_types=(
            ErrorType.PROVIDER_ERROR,
            ErrorType.OVERLOADED,
            ErrorType.OVERLOADED,
        ),
        current_error_types=(
            ErrorType.PROVIDER_ERROR,
            ErrorType.UNKNOWN_ERROR,
            ErrorType.RESPONSE_MODE_MISMATCH,
        ),
    )

    assert comparison.baseline == {
        "provider_error": 1,
        "overloaded": 2,
        "unknown_error": 0,
        "response_mode_mismatch": 0,
    }
    assert comparison.current == {
        "provider_error": 1,
        "overloaded": 0,
        "unknown_error": 1,
        "response_mode_mismatch": 1,
    }
    assert comparison.delta == {
        "provider_error": 0,
        "overloaded": -2,
        "unknown_error": 1,
        "response_mode_mismatch": 1,
    }


def test_empty_countsは4種類すべてをゼロで返す() -> None:
    assert empty_error_type_counts() == {
        "provider_error": 0,
        "overloaded": 0,
        "unknown_error": 0,
        "response_mode_mismatch": 0,
    }


def test不正なerror_typeはvalue_errorになる() -> None:
    with pytest.raises(ValueError):
        count_error_types(("invalid_error_type",))
