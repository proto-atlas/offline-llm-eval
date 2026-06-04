from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from offline_llm_eval.api.database import get_or_create_session_factory
from offline_llm_eval.api.error_response import ApiErrorCode, build_error_response
from offline_llm_eval.dataset.repository import Dataset, JsonObject, JsonValue
from offline_llm_eval.run.case_result import CaseResultRecord
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus
from offline_llm_eval.run.metrics import (
    RunMetrics,
    calculate_run_metrics,
    resolve_effective_case_status,
)
from offline_llm_eval.run.usage import resolve_usage_summary
from offline_llm_eval.runner.error_type import ErrorType

router = APIRouter(prefix="/api/runs")
DEFAULT_RUN_LIST_LIMIT: Final = 100
MAX_RUN_LIST_LIMIT: Final = 1000


async def get_database_session(request: Request) -> AsyncIterator[AsyncSession]:
    session_factory = get_or_create_session_factory(request.app)
    async with session_factory() as session:
        yield session


@router.get("", response_model=None)
async def list_runs(
    session: Annotated[AsyncSession, Depends(get_database_session)],
    target_label: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_RUN_LIST_LIMIT)] = DEFAULT_RUN_LIST_LIMIT,
) -> JsonObject:
    runs = await build_run_list(session, target_label=target_label, limit=limit)
    return {
        "runs": runs,
        "target_label": target_label,
        "limit": limit,
    }


@router.get("/{run_id}", response_model=None)
async def get_run_summary(
    run_id: int,
    session: Annotated[AsyncSession, Depends(get_database_session)],
) -> JsonObject | JSONResponse:
    summary = await build_run_summary(session, run_id)
    if summary is None:
        return build_error_response(
            ApiErrorCode.RUN_NOT_FOUND,
            "実行が見つかりません。",
            {"run_id": run_id},
        )

    return summary


async def build_run_summary(session: AsyncSession, run_id: int) -> JsonObject | None:
    run_and_dataset = await _load_run_and_dataset(session, run_id)
    if run_and_dataset is None:
        return None

    run, dataset = run_and_dataset
    case_statuses = await _load_case_statuses(session, run_id)
    metrics = calculate_run_metrics(case_statuses)
    error_type_counts = await _load_error_type_counts(session, run_id)
    usage = resolve_usage_summary(None)

    return {
        "run": {
            "run_id": run.run_id,
            "dataset_id": run.dataset_id,
            "dataset_name": dataset.name,
            "dataset_version": dataset.dataset_version,
            "target_label": run.target_label,
            "target_version": run.target_version,
            "status": RunStatus(run.status).value,
            "started_at": _datetime_to_json(run.started_at),
            "completed_at": _datetime_to_json(run.completed_at),
            "last_heartbeat_at": _datetime_to_json(run.last_heartbeat_at),
        },
        "counts": _counts_to_json(metrics),
        "rates": _rates_to_json(metrics),
        "error_types": error_type_counts,
        "usage": {
            "usage_source": usage.usage_source.value,
            "usage_json": usage.usage_json,
        },
        "gate": {
            "config_snapshot": run.gate_config_snapshot_json,
            "result": run.gate_result_json,
        },
    }


async def build_run_list(
    session: AsyncSession,
    *,
    target_label: str | None,
    limit: int,
) -> list[JsonValue]:
    statement = select(RunRecord, Dataset).join(
        Dataset,
        RunRecord.dataset_id == Dataset.dataset_id,
    )
    if target_label is not None:
        statement = statement.where(RunRecord.target_label == target_label)

    statement = statement.order_by(RunRecord.created_at.desc()).limit(limit)
    result = await session.execute(statement)
    return [_run_list_item_to_json(run, dataset) for run, dataset in result.all()]


async def _load_run_and_dataset(
    session: AsyncSession,
    run_id: int,
) -> tuple[RunRecord, Dataset] | None:
    result = await session.execute(
        select(RunRecord, Dataset)
        .join(Dataset, RunRecord.dataset_id == Dataset.dataset_id)
        .where(RunRecord.run_id == run_id)
    )
    row = result.one_or_none()
    if row is None:
        return None

    return row[0], row[1]


async def _load_case_statuses(session: AsyncSession, run_id: int) -> tuple[str, ...]:
    result = await session.execute(
        select(CaseResultRecord.status, CaseResultRecord.final_status).where(
            CaseResultRecord.run_id == run_id
        )
    )
    return tuple(
        resolve_effective_case_status(status, final_status).value
        for status, final_status in result.all()
    )


async def _load_error_type_counts(session: AsyncSession, run_id: int) -> JsonObject:
    counts = _empty_error_type_counts()
    result = await session.execute(
        select(CaseResultRecord.error_type, func.count(CaseResultRecord.case_result_id))
        .where(
            CaseResultRecord.run_id == run_id,
            CaseResultRecord.error_type.is_not(None),
        )
        .group_by(CaseResultRecord.error_type)
    )

    for error_type_value, count in result.all():
        if error_type_value is None:
            continue
        counts[ErrorType(error_type_value).value] = int(count)

    return counts


def _run_list_item_to_json(run: RunRecord, dataset: Dataset) -> JsonObject:
    return {
        "run_id": run.run_id,
        "dataset_id": run.dataset_id,
        "dataset_name": dataset.name,
        "dataset_version": dataset.dataset_version,
        "target_label": run.target_label,
        "target_version": run.target_version,
        "status": RunStatus(run.status).value,
        "started_at": _datetime_to_json(run.started_at),
        "completed_at": _datetime_to_json(run.completed_at),
        "last_heartbeat_at": _datetime_to_json(run.last_heartbeat_at),
        "created_at": _datetime_to_json(run.created_at),
    }


def _empty_error_type_counts() -> JsonObject:
    return {error_type.value: 0 for error_type in ErrorType}


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


def _datetime_to_json(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()
