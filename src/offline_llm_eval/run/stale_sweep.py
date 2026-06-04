from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from offline_llm_eval.run.heartbeat import RunRecord, RunStatus, utc_now

DEFAULT_STALE_TIMEOUT: Final = timedelta(hours=1)


@dataclass(frozen=True, slots=True)
class StaleSweepResult:
    aborted_run_ids: tuple[int, ...]


async def sweep_stale_runs_for_run_start(
    session: AsyncSession,
    *,
    dataset_id: int,
    target_label: str,
    now: datetime | None = None,
    stale_timeout: timedelta = DEFAULT_STALE_TIMEOUT,
) -> StaleSweepResult:
    return await abort_stale_runs(
        session,
        now=now,
        stale_timeout=stale_timeout,
        dataset_id=dataset_id,
        target_label=target_label,
    )


async def sweep_stale_runs_for_dataset_import(
    session: AsyncSession,
    *,
    dataset_id: int,
    now: datetime | None = None,
    stale_timeout: timedelta = DEFAULT_STALE_TIMEOUT,
) -> StaleSweepResult:
    return await abort_stale_runs(
        session,
        now=now,
        stale_timeout=stale_timeout,
        dataset_id=dataset_id,
    )


async def sweep_stale_runs_for_abort_stale(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    stale_timeout: timedelta = DEFAULT_STALE_TIMEOUT,
) -> StaleSweepResult:
    return await abort_stale_runs(session, now=now, stale_timeout=stale_timeout)


async def abort_stale_runs(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    stale_timeout: timedelta = DEFAULT_STALE_TIMEOUT,
    dataset_id: int | None = None,
    target_label: str | None = None,
) -> StaleSweepResult:
    sweep_at = now or utc_now()
    cutoff = sweep_at - stale_timeout
    stale_at = func.coalesce(RunRecord.last_heartbeat_at, RunRecord.started_at)
    query = select(RunRecord.run_id).where(
        RunRecord.status == RunStatus.RUNNING.value,
        stale_at < cutoff,
    )
    if dataset_id is not None:
        query = query.where(RunRecord.dataset_id == dataset_id)
    if target_label is not None:
        query = query.where(RunRecord.target_label == target_label)

    result = await session.execute(query.order_by(RunRecord.run_id))
    run_ids = tuple(result.scalars().all())
    if not run_ids:
        return StaleSweepResult(aborted_run_ids=())

    await session.execute(
        update(RunRecord)
        .where(RunRecord.run_id.in_(run_ids))
        .values(
            status=RunStatus.ABORTED.value,
            completed_at=sweep_at,
        )
    )
    await session.flush()
    return StaleSweepResult(aborted_run_ids=run_ids)
