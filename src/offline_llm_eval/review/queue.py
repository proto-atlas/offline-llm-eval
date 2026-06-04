from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.run.case_result import CaseResultRecord
from offline_llm_eval.run.repository import RunRepository, RunSnapshot


@dataclass(frozen=True, slots=True)
class ReviewQueueItem:
    case_key: str
    status: CaseResultStatus
    reviewer_verdict: str | None
    reviewer_note: str | None
    reviewed_at: datetime | None
    final_status: str | None


@dataclass(frozen=True, slots=True)
class ReviewQueue:
    run: RunSnapshot
    items: tuple[ReviewQueueItem, ...]


async def get_review_queue(session: AsyncSession, run_id: int) -> ReviewQueue | None:
    run = await RunRepository(session).get_run(run_id)
    if run is None:
        return None

    return ReviewQueue(
        run=run,
        items=await _load_review_queue_items(session, run_id=run_id),
    )


async def _load_review_queue_items(
    session: AsyncSession,
    *,
    run_id: int,
) -> tuple[ReviewQueueItem, ...]:
    result = await session.execute(
        select(
            CaseResultRecord.case_key,
            CaseResultRecord.status,
            CaseResultRecord.reviewer_verdict,
            CaseResultRecord.reviewer_note,
            CaseResultRecord.reviewed_at,
            CaseResultRecord.final_status,
        )
        .where(
            CaseResultRecord.run_id == run_id,
            CaseResultRecord.status == CaseResultStatus.NEEDS_REVIEW.value,
        )
        .order_by(CaseResultRecord.case_key)
    )
    items = tuple(
        ReviewQueueItem(
            case_key=str(case_key),
            status=CaseResultStatus(str(status)),
            reviewer_verdict=None if reviewer_verdict is None else str(reviewer_verdict),
            reviewer_note=None if reviewer_note is None else str(reviewer_note),
            reviewed_at=reviewed_at,
            final_status=None if final_status is None else str(final_status),
        )
        for (
            case_key,
            status,
            reviewer_verdict,
            reviewer_note,
            reviewed_at,
            final_status,
        ) in result.all()
    )
    return tuple(sorted(items, key=_review_queue_sort_key))


def _review_queue_sort_key(item: ReviewQueueItem) -> tuple[bool, str]:
    return (item.reviewer_verdict is not None, item.case_key)
