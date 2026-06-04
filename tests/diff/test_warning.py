from datetime import datetime

from offline_llm_eval.diff.warning import (
    CaseUpdatedAt,
    DiffWarning,
    DiffWarningCode,
    collect_case_updated_at_warnings,
)
from offline_llm_eval.run.heartbeat import RunStatus
from offline_llm_eval.run.repository import RunSnapshot

BASELINE_STARTED_AT = datetime(2026, 5, 27, 10, 0, 0)
BEFORE_BASELINE = datetime(2026, 5, 27, 9, 59, 59)
SAME_AS_BASELINE = datetime(2026, 5, 27, 10, 0, 0)
AFTER_BASELINE = datetime(2026, 5, 27, 10, 0, 1)


def test_case_updated_atがbaseline開始後ならwarningを返す() -> None:
    warnings = collect_case_updated_at_warnings(
        baseline_run=baseline_run(),
        cases=(CaseUpdatedAt("case_changed", AFTER_BASELINE),),
    )

    assert warnings == (
        DiffWarning(
            code=DiffWarningCode.CASE_UPDATED_AFTER_BASELINE,
            case_key="case_changed",
            baseline_run_id=7,
            baseline_started_at=BASELINE_STARTED_AT,
            case_updated_at=AFTER_BASELINE,
        ),
    )


def test_case_updated_atがbaseline開始以前ならwarningを返さない() -> None:
    warnings = collect_case_updated_at_warnings(
        baseline_run=baseline_run(),
        cases=(
            CaseUpdatedAt("case_before", BEFORE_BASELINE),
            CaseUpdatedAt("case_same", SAME_AS_BASELINE),
        ),
    )

    assert warnings == ()


def test_warningはcase_key順で返す() -> None:
    warnings = collect_case_updated_at_warnings(
        baseline_run=baseline_run(),
        cases=(
            CaseUpdatedAt("case_c", AFTER_BASELINE),
            CaseUpdatedAt("case_a", AFTER_BASELINE),
            CaseUpdatedAt("case_b", BEFORE_BASELINE),
        ),
    )

    assert [warning.case_key for warning in warnings] == ["case_a", "case_c"]


def baseline_run() -> RunSnapshot:
    return RunSnapshot(
        run_id=7,
        dataset_id=1,
        target_label="local",
        target_version="mock-v1",
        status=RunStatus.COMPLETED,
        started_at=BASELINE_STARTED_AT,
        completed_at=datetime(2026, 5, 27, 10, 5, 0),
        last_heartbeat_at=datetime(2026, 5, 27, 10, 4, 30),
        gate_config_snapshot_json=None,
        gate_result_json=None,
    )
