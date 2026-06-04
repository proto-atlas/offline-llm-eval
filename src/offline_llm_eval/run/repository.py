from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from offline_llm_eval.dataset.repository import JsonObject
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus, utc_now


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    run_id: int
    dataset_id: int
    target_label: str
    target_version: str | None
    status: RunStatus
    started_at: datetime
    completed_at: datetime | None
    last_heartbeat_at: datetime
    gate_config_snapshot_json: JsonObject | None
    gate_result_json: JsonObject | None


class RunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_run(
        self,
        *,
        dataset_id: int,
        target_label: str,
        target_version: str | None = None,
        gate_config_snapshot_json: JsonObject | None = None,
        started_at: datetime | None = None,
    ) -> RunSnapshot:
        run = RunRecord(
            dataset_id=dataset_id,
            target_label=target_label,
            target_version=target_version,
            status=RunStatus.RUNNING.value,
            gate_config_snapshot_json=_copy_json_object(gate_config_snapshot_json),
        )
        if started_at is not None:
            run.started_at = started_at

        self._session.add(run)
        await self._session.flush()
        await self._session.refresh(run)
        return _to_snapshot(run)

    async def get_run(self, run_id: int) -> RunSnapshot | None:
        run = await self._get_run_record(run_id)
        if run is None:
            return None
        return _to_snapshot(run)

    async def complete_run(
        self,
        run_id: int,
        *,
        completed_at: datetime | None = None,
        gate_result_json: JsonObject | None = None,
    ) -> RunSnapshot | None:
        return await self._finish_running_run(
            run_id,
            status=RunStatus.COMPLETED,
            completed_at=completed_at,
            gate_result_json=gate_result_json,
        )

    async def abort_run(
        self,
        run_id: int,
        *,
        completed_at: datetime | None = None,
    ) -> RunSnapshot | None:
        return await self._finish_running_run(
            run_id,
            status=RunStatus.ABORTED,
            completed_at=completed_at,
            gate_result_json=None,
        )

    async def _finish_running_run(
        self,
        run_id: int,
        *,
        status: RunStatus,
        completed_at: datetime | None,
        gate_result_json: JsonObject | None,
    ) -> RunSnapshot | None:
        run = await self._get_run_record(run_id)
        if run is None or run.status != RunStatus.RUNNING.value:
            return None

        finished_at = completed_at or utc_now()
        run.status = status.value
        run.completed_at = finished_at
        run.updated_at = finished_at
        if status is RunStatus.COMPLETED:
            run.gate_result_json = _copy_json_object(gate_result_json)

        await self._session.flush()
        return _to_snapshot(run)

    async def _get_run_record(self, run_id: int) -> RunRecord | None:
        result = await self._session.execute(select(RunRecord).where(RunRecord.run_id == run_id))
        return result.scalar_one_or_none()


def _copy_json_object(value: JsonObject | None) -> JsonObject | None:
    if value is None:
        return None
    return dict(value)


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
