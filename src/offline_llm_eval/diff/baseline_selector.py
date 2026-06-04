from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from offline_llm_eval.run.heartbeat import RunRecord, RunStatus
from offline_llm_eval.run.repository import RunSnapshot

MAIN_TARGET_LABEL: Final = "main"
RUN_SPEC_PREFIX: Final = "run:"


class BaselineWarning(StrEnum):
    BASELINE_ABORTED = "baseline_aborted"


@dataclass(frozen=True, slots=True)
class BaselineSelection:
    run: RunSnapshot
    warnings: tuple[BaselineWarning, ...] = ()


class BaselineSelectorError(ValueError):
    code = "baseline_selector_error"


class BaselineNotFoundError(BaselineSelectorError):
    code = "baseline_not_found"

    def __init__(self, baseline_spec: str) -> None:
        self.baseline_spec = baseline_spec
        super().__init__(f"{self.code}: 基準実行が見つかりません: baseline={baseline_spec}")


class BaselineInProgressError(BaselineSelectorError):
    code = "baseline_in_progress"
    exit_code = 1

    def __init__(self, run_id: int) -> None:
        self.run_id = run_id
        super().__init__(
            f"{self.code}: 基準実行が実行中のため差分を作成できません: run_id={run_id}"
        )


class InvalidBaselineSpecError(BaselineSelectorError):
    code = "validation_error"

    def __init__(self, baseline_spec: str) -> None:
        self.baseline_spec = baseline_spec
        super().__init__(f"{self.code}: 基準実行の指定が不正です: {baseline_spec}")


async def select_baseline_run(
    session: AsyncSession,
    *,
    current_run: RunSnapshot,
    baseline_spec: str,
) -> BaselineSelection:
    if baseline_spec == "latest":
        return await _select_latest(
            session,
            current_run=current_run,
            target_label=current_run.target_label,
            baseline_spec=baseline_spec,
        )

    if baseline_spec == "latest_main":
        return await _select_latest(
            session,
            current_run=current_run,
            target_label=MAIN_TARGET_LABEL,
            baseline_spec=baseline_spec,
        )

    if baseline_spec.startswith(RUN_SPEC_PREFIX):
        run_id = _parse_run_id(baseline_spec)
        return await _select_explicit_run(session, run_id=run_id, baseline_spec=baseline_spec)

    raise InvalidBaselineSpecError(baseline_spec)


async def _select_latest(
    session: AsyncSession,
    *,
    current_run: RunSnapshot,
    target_label: str,
    baseline_spec: str,
) -> BaselineSelection:
    result = await session.execute(
        select(RunRecord)
        .where(
            RunRecord.dataset_id == current_run.dataset_id,
            RunRecord.target_label == target_label,
            RunRecord.status == RunStatus.COMPLETED.value,
            RunRecord.run_id != current_run.run_id,
        )
        .order_by(RunRecord.created_at.desc(), RunRecord.run_id.desc())
        .limit(1)
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise BaselineNotFoundError(baseline_spec)
    return BaselineSelection(run=_to_snapshot(run))


async def _select_explicit_run(
    session: AsyncSession,
    *,
    run_id: int,
    baseline_spec: str,
) -> BaselineSelection:
    result = await session.execute(select(RunRecord).where(RunRecord.run_id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        raise BaselineNotFoundError(baseline_spec)

    status = RunStatus(run.status)
    if status is RunStatus.RUNNING:
        raise BaselineInProgressError(run_id)

    warnings: tuple[BaselineWarning, ...] = ()
    if status is RunStatus.ABORTED:
        warnings = (BaselineWarning.BASELINE_ABORTED,)

    return BaselineSelection(run=_to_snapshot(run), warnings=warnings)


def _parse_run_id(baseline_spec: str) -> int:
    raw_run_id = baseline_spec.removeprefix(RUN_SPEC_PREFIX)
    if raw_run_id == "":
        raise InvalidBaselineSpecError(baseline_spec)
    try:
        run_id = int(raw_run_id)
    except ValueError as error:
        raise InvalidBaselineSpecError(baseline_spec) from error
    if run_id <= 0:
        raise InvalidBaselineSpecError(baseline_spec)
    return run_id


def _to_snapshot(run: RunRecord) -> RunSnapshot:
    return RunSnapshot(
        run_id=run.run_id,
        dataset_id=run.dataset_id,
        target_label=run.target_label,
        target_version=run.target_version,
        status=RunStatus(run.status),
        started_at=run.started_at,
        completed_at=run.completed_at,
        last_heartbeat_at=run.last_heartbeat_at,
        gate_config_snapshot_json=run.gate_config_snapshot_json,
        gate_result_json=run.gate_result_json,
    )
