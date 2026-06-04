import pytest

from offline_llm_eval.diff.comparator import (
    CaseResultForComparison,
    compare_case_keys,
    compare_runs,
)
from offline_llm_eval.evaluator.status import CaseResultStatus


def test_case_key集合を突合する() -> None:
    comparison = compare_case_keys(
        baseline_case_keys=("case_c", "case_a", "case_b"),
        current_case_keys=("case_d", "case_b", "case_c"),
    )

    assert comparison.shared_case_keys == ("case_b", "case_c")
    assert comparison.added_case_keys == ("case_d",)
    assert comparison.removed_case_keys == ("case_a",)


def test_baselineとcurrentの集計差分を返す() -> None:
    comparison = compare_runs(
        baseline_cases=(
            CaseResultForComparison("case_pass", CaseResultStatus.PASS),
            CaseResultForComparison("case_failed", CaseResultStatus.FAILED),
            CaseResultForComparison("case_skipped_1", CaseResultStatus.SKIPPED),
            CaseResultForComparison("case_skipped_2", CaseResultStatus.SKIPPED),
        ),
        current_cases=(
            CaseResultForComparison("case_pass_1", CaseResultStatus.PASS),
            CaseResultForComparison("case_pass_2", CaseResultStatus.PASS),
            CaseResultForComparison("case_failed", CaseResultStatus.FAILED),
            CaseResultForComparison("case_skipped", CaseResultStatus.SKIPPED),
        ),
    )

    assert comparison.metrics.baseline.total_count == 4
    assert comparison.metrics.baseline.executed_count == 2
    assert comparison.metrics.current.total_count == 4
    assert comparison.metrics.current.executed_count == 3
    assert comparison.metrics.delta.total_count == 0
    assert comparison.metrics.delta.passed_count == 1
    assert comparison.metrics.delta.failed_count == 0
    assert comparison.metrics.delta.needs_review_count == 0
    assert comparison.metrics.delta.skipped_count == -1
    assert comparison.metrics.delta.executed_count == 1
    assert comparison.metrics.delta.pass_rate == pytest.approx(1 / 6)
    assert comparison.metrics.delta.fail_rate == pytest.approx(-1 / 6)
    assert comparison.metrics.delta.skipped_ratio == pytest.approx(-0.25)


def test文字列statusも既存metricsと同じように扱う() -> None:
    comparison = compare_runs(
        baseline_cases=(CaseResultForComparison("case_a", "pass"),),
        current_cases=(
            CaseResultForComparison("case_a", "needs_review"),
            CaseResultForComparison("case_b", "skipped"),
        ),
    )

    assert comparison.case_keys.shared_case_keys == ("case_a",)
    assert comparison.case_keys.added_case_keys == ("case_b",)
    assert comparison.case_keys.removed_case_keys == ()
    assert comparison.metrics.current.needs_review_count == 1
    assert comparison.metrics.current.skipped_count == 1
