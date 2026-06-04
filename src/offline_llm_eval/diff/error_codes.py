from collections.abc import Sequence
from dataclasses import dataclass

from offline_llm_eval.runner.error_type import ErrorType

type ErrorTypeValue = ErrorType | str | None
type ErrorTypeCounts = dict[str, int]


@dataclass(frozen=True, slots=True)
class ErrorTypeCountComparison:
    baseline: ErrorTypeCounts
    current: ErrorTypeCounts
    delta: ErrorTypeCounts


def count_error_types(error_types: Sequence[ErrorTypeValue]) -> ErrorTypeCounts:
    counts = empty_error_type_counts()
    for error_type in error_types:
        if error_type is None:
            continue
        counts[ErrorType(error_type).value] += 1
    return counts


def compare_error_type_counts(
    *,
    baseline_error_types: Sequence[ErrorTypeValue],
    current_error_types: Sequence[ErrorTypeValue],
) -> ErrorTypeCountComparison:
    baseline = count_error_types(baseline_error_types)
    current = count_error_types(current_error_types)
    return ErrorTypeCountComparison(
        baseline=baseline,
        current=current,
        delta={
            error_type.value: current[error_type.value] - baseline[error_type.value]
            for error_type in ErrorType
        },
    )


def empty_error_type_counts() -> ErrorTypeCounts:
    return {error_type.value: 0 for error_type in ErrorType}
