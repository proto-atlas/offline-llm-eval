import pytest

from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.run.metrics import RunMetrics, calculate_run_metrics


def test_case_statusからcountsとratesを計算する() -> None:
    metrics = calculate_run_metrics(
        (
            CaseResultStatus.PASS,
            CaseResultStatus.PASS,
            CaseResultStatus.FAILED,
            CaseResultStatus.NEEDS_REVIEW,
            CaseResultStatus.SKIPPED,
        )
    )

    assert metrics == RunMetrics(
        total_count=5,
        passed_count=2,
        failed_count=1,
        needs_review_count=1,
        skipped_count=1,
        executed_count=4,
        pass_rate=0.5,
        fail_rate=0.25,
        skipped_ratio=0.2,
    )


def test_db由来の文字列statusも計算できる() -> None:
    metrics = calculate_run_metrics(("pass", "failed", "skipped"))

    assert metrics.passed_count == 1
    assert metrics.failed_count == 1
    assert metrics.skipped_count == 1
    assert metrics.executed_count == 2
    assert metrics.pass_rate == 0.5
    assert metrics.fail_rate == 0.5
    assert metrics.skipped_ratio == pytest.approx(1 / 3)


def test全skippedならexecuted_countは0でskipped_ratioは1になる() -> None:
    metrics = calculate_run_metrics((CaseResultStatus.SKIPPED, CaseResultStatus.SKIPPED))

    assert metrics.executed_count == 0
    assert metrics.pass_rate == 0.0
    assert metrics.fail_rate == 0.0
    assert metrics.skipped_ratio == 1.0


def test空なら全rateを0にする() -> None:
    metrics = calculate_run_metrics(())

    assert metrics == RunMetrics(
        total_count=0,
        passed_count=0,
        failed_count=0,
        needs_review_count=0,
        skipped_count=0,
        executed_count=0,
        pass_rate=0.0,
        fail_rate=0.0,
        skipped_ratio=0.0,
    )


def test未知のstatusは拒否する() -> None:
    with pytest.raises(ValueError):
        calculate_run_metrics(("unknown",))
