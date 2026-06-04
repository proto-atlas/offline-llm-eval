from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from offline_llm_eval.api.error_response import ApiErrorCode, build_error_response
from offline_llm_eval.api.runs import get_database_session
from offline_llm_eval.dataset.repository import JsonObject, JsonValue
from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.evidence.sanitize import (
    REVIEWER_VERDICT_MAX_LENGTH,
    sanitize_optional_reviewer_verdict,
    sanitize_reviewer_note,
    sanitize_reviewer_verdict,
)
from offline_llm_eval.review.queue import ReviewQueue, ReviewQueueItem, get_review_queue
from offline_llm_eval.run.case_result import CaseResultRecord
from offline_llm_eval.run.heartbeat import RunRecord
from offline_llm_eval.run.metrics import (
    RunMetrics,
    calculate_run_metrics,
    resolve_effective_case_status,
)

router = APIRouter(prefix="/api/runs")


class ReviewFinalStatus(StrEnum):
    PASS = "pass"
    FAILED = "failed"


class ReviewPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer_verdict: str = Field(min_length=1, max_length=REVIEWER_VERDICT_MAX_LENGTH)
    final_status: ReviewFinalStatus
    reviewer_note: str | None = Field(default=None, max_length=4000)


@dataclass(frozen=True, slots=True)
class ReviewedCase:
    case_result_id: int
    case_key: str
    status: CaseResultStatus
    reviewer_verdict: str
    reviewer_note: str | None
    reviewed_at: datetime
    final_status: ReviewFinalStatus
    effective_status: CaseResultStatus


@dataclass(frozen=True, slots=True)
class ReviewRecomputeResult:
    case: ReviewedCase
    metrics: RunMetrics


class ReviewRunNotFoundError(ValueError):
    def __init__(self, run_id: int) -> None:
        super().__init__(f"実行が見つかりません: run_id={run_id}")
        self.run_id = run_id


class ReviewCaseNotFoundError(ValueError):
    def __init__(self, run_id: int, case_key: str) -> None:
        super().__init__(f"ケースが見つかりません: run_id={run_id}, case_key={case_key}")
        self.run_id = run_id
        self.case_key = case_key


@router.get("/{run_id}/review-queue", response_model=None)
async def get_run_review_queue(
    run_id: int,
    session: Annotated[AsyncSession, Depends(get_database_session)],
) -> JsonObject | JSONResponse:
    queue = await get_review_queue(session, run_id)
    if queue is None:
        return build_error_response(
            ApiErrorCode.RUN_NOT_FOUND,
            "実行が見つかりません。",
            {"run_id": run_id},
        )
    return _review_queue_to_json(queue)


@router.patch("/{run_id}/cases/{case_key}/review", response_model=None)
async def patch_case_review(
    run_id: int,
    case_key: str,
    payload: ReviewPatchRequest,
    session: Annotated[AsyncSession, Depends(get_database_session)],
) -> JsonObject | JSONResponse:
    try:
        result = await apply_review_transaction(
            session,
            run_id=run_id,
            case_key=case_key,
            reviewer_verdict=payload.reviewer_verdict,
            final_status=payload.final_status,
            reviewer_note=payload.reviewer_note,
        )
    except ReviewRunNotFoundError as error:
        return build_error_response(
            ApiErrorCode.RUN_NOT_FOUND,
            "実行が見つかりません。",
            {"run_id": error.run_id},
        )
    except ReviewCaseNotFoundError as error:
        return build_error_response(
            ApiErrorCode.CASE_NOT_FOUND,
            "ケースが見つかりません。",
            {"run_id": error.run_id, "case_key": error.case_key},
        )

    return _review_result_to_json(result)


async def apply_review_transaction(
    session: AsyncSession,
    *,
    run_id: int,
    case_key: str,
    reviewer_verdict: str,
    final_status: ReviewFinalStatus,
    reviewer_note: str | None,
    reviewed_at: datetime | None = None,
) -> ReviewRecomputeResult:
    sanitized_reviewer_verdict = sanitize_reviewer_verdict(reviewer_verdict)
    sanitized_reviewer_note = sanitize_reviewer_note(reviewer_note)
    async with session.begin():
        case = await _load_case_result(session, run_id=run_id, case_key=case_key)
        if case is None:
            await _raise_review_not_found(session, run_id=run_id, case_key=case_key)

        review_timestamp = reviewed_at or datetime.now(UTC)
        case.reviewer_verdict = sanitized_reviewer_verdict
        case.reviewer_note = sanitized_reviewer_note
        case.reviewed_at = review_timestamp
        case.final_status = final_status.value
        await session.flush()

        metrics = calculate_run_metrics(await _load_effective_case_statuses(session, run_id))
        return ReviewRecomputeResult(
            case=ReviewedCase(
                case_result_id=case.case_result_id,
                case_key=case.case_key,
                status=CaseResultStatus(case.status),
                reviewer_verdict=sanitized_reviewer_verdict,
                reviewer_note=sanitized_reviewer_note,
                reviewed_at=review_timestamp,
                final_status=final_status,
                effective_status=resolve_effective_case_status(case.status, case.final_status),
            ),
            metrics=metrics,
        )


async def _load_case_result(
    session: AsyncSession,
    *,
    run_id: int,
    case_key: str,
) -> CaseResultRecord | None:
    result = await session.execute(
        select(CaseResultRecord).where(
            CaseResultRecord.run_id == run_id,
            CaseResultRecord.case_key == case_key,
        )
    )
    return result.scalar_one_or_none()


async def _raise_review_not_found(
    session: AsyncSession,
    *,
    run_id: int,
    case_key: str,
) -> NoReturn:
    if not await _run_exists(session, run_id):
        raise ReviewRunNotFoundError(run_id)
    raise ReviewCaseNotFoundError(run_id, case_key)


async def _run_exists(session: AsyncSession, run_id: int) -> bool:
    result = await session.execute(select(RunRecord.run_id).where(RunRecord.run_id == run_id))
    return result.scalar_one_or_none() is not None


async def _load_effective_case_statuses(
    session: AsyncSession,
    run_id: int,
) -> tuple[CaseResultStatus, ...]:
    result = await session.execute(
        select(CaseResultRecord.status, CaseResultRecord.final_status).where(
            CaseResultRecord.run_id == run_id
        )
    )
    return tuple(
        resolve_effective_case_status(status, final_status) for status, final_status in result.all()
    )


def _review_queue_to_json(queue: ReviewQueue) -> JsonObject:
    return {
        "run": {
            "run_id": queue.run.run_id,
            "dataset_id": queue.run.dataset_id,
            "target_label": queue.run.target_label,
            "target_version": queue.run.target_version,
            "status": queue.run.status.value,
        },
        "items": [_review_queue_item_to_json(item) for item in queue.items],
    }


def _review_queue_item_to_json(item: ReviewQueueItem) -> JsonObject:
    return {
        "case_key": item.case_key,
        "status": item.status.value,
        "reviewer_verdict": sanitize_optional_reviewer_verdict(item.reviewer_verdict),
        "reviewer_note": sanitize_reviewer_note(item.reviewer_note),
        "reviewed_at": _datetime_to_json(item.reviewed_at),
        "final_status": item.final_status,
    }


def _review_result_to_json(result: ReviewRecomputeResult) -> JsonObject:
    return {
        "case": {
            "case_result_id": result.case.case_result_id,
            "case_key": result.case.case_key,
            "status": result.case.status.value,
            "effective_status": result.case.effective_status.value,
        },
        "review": {
            "reviewer_verdict": result.case.reviewer_verdict,
            "reviewer_note": result.case.reviewer_note,
            "reviewed_at": result.case.reviewed_at.isoformat(),
            "final_status": result.case.final_status.value,
        },
        "counts": _counts_to_json(result.metrics),
        "rates": _rates_to_json(result.metrics),
    }


def _counts_to_json(metrics: RunMetrics) -> JsonObject:
    return {
        "total": metrics.total_count,
        "executed": metrics.executed_count,
        "passed": metrics.passed_count,
        "failed": metrics.failed_count,
        "needs_review": metrics.needs_review_count,
        "skipped": metrics.skipped_count,
    }


def _rates_to_json(metrics: RunMetrics) -> JsonObject:
    return {
        "pass_rate": metrics.pass_rate,
        "fail_rate": metrics.fail_rate,
        "skipped_ratio": metrics.skipped_ratio,
    }


def _datetime_to_json(value: datetime | None) -> JsonValue:
    if value is None:
        return None
    return value.isoformat()
