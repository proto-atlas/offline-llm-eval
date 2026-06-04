import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, cast

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Integer, String, text, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from offline_llm_eval.dataset.repository import JsonObject
from offline_llm_eval.db.base import Base

HEARTBEAT_INTERVAL_SECONDS: Final = 30.0

type Sleep = Callable[[float], Awaitable[None]]


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"


class RunRecord(Base):
    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint(
            "status in ('running', 'completed', 'aborted')",
            name="ck_runs_status",
        ),
    )

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.dataset_id", ondelete="RESTRICT"),
        nullable=False,
    )
    target_label: Mapped[str] = mapped_column(String(255), nullable=False)
    target_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_heartbeat_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
    gate_config_snapshot_json: Mapped[JsonObject | None] = mapped_column(JSON, nullable=True)
    gate_result_json: Mapped[JsonObject | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


def utc_now() -> datetime:
    return datetime.now(UTC)


async def update_run_heartbeat(
    session_factory: async_sessionmaker[AsyncSession],
    run_id: int,
    *,
    now: datetime | None = None,
) -> bool:
    heartbeat_at = now or utc_now()
    async with session_factory() as session:
        result = await session.execute(
            update(RunRecord)
            .where(
                RunRecord.run_id == run_id,
                RunRecord.status == RunStatus.RUNNING.value,
            )
            .values(last_heartbeat_at=heartbeat_at)
        )
        await session.commit()

    return _rowcount(result) == 1


async def run_heartbeat_loop(
    session_factory: async_sessionmaker[AsyncSession],
    run_id: int,
    stop_event: asyncio.Event,
    *,
    interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
    sleep: Sleep = asyncio.sleep,
) -> None:
    while not stop_event.is_set():
        await update_run_heartbeat(session_factory, run_id)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except TimeoutError:
            await sleep(0)


def _rowcount(result: object) -> int:
    cursor_result = cast(CursorResult[object], result)
    return cursor_result.rowcount or 0
