from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, cast

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import Boolean, DateTime, Integer, String, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from offline_llm_eval.api.error_response import ApiErrorCode, build_error_response
from offline_llm_eval.api.runs import get_database_session
from offline_llm_eval.dataset.assertion_model import AssertionRecord
from offline_llm_eval.dataset.repository import JsonObject, JsonValue
from offline_llm_eval.diff.baseline_selector import (
    BaselineInProgressError,
    BaselineNotFoundError,
    BaselineSelection,
    BaselineWarning,
    InvalidBaselineSpecError,
    select_baseline_run,
)
from offline_llm_eval.diff.classification import (
    AssertionForDiff,
    CaseForDiff,
    DiffClassification,
    classify_diff,
)
from offline_llm_eval.diff.comparator import (
    BaselineComparison,
    CaseResultForComparison,
    compare_runs,
)
from offline_llm_eval.diff.error_codes import (
    ErrorTypeCountComparison,
    compare_error_type_counts,
)
from offline_llm_eval.diff.warning import (
    CaseUpdatedAt,
    DiffWarning,
    collect_case_updated_at_warnings,
)
from offline_llm_eval.evidence.sanitize import sanitize_evidence_text
from offline_llm_eval.run.case_result import CaseResultRecord
from offline_llm_eval.run.metrics import RunMetrics
from offline_llm_eval.run.repository import RunRepository, RunSnapshot

router = APIRouter(prefix="/api/runs")


@dataclass(frozen=True, slots=True)
class CaseResultForDiff:
    case_key: str
    status: str
    error_type: str | None


@dataclass(frozen=True, slots=True)
class DatasetCaseForDiff:
    case_id: int
    case_key: str
    is_active: bool
    updated_at: datetime


@router.get("/{run_id}/diff", response_model=None)
async def get_run_diff(
    run_id: int,
    session: Annotated[AsyncSession, Depends(get_database_session)],
    baseline: str = "latest",
) -> JsonObject | JSONResponse:
    current_run = await RunRepository(session).get_run(run_id)
    if current_run is None:
        return build_error_response(
            ApiErrorCode.RUN_NOT_FOUND,
            "実行が見つかりません。",
            {"run_id": run_id},
        )

    try:
        baseline_selection = await select_baseline_run(
            session,
            current_run=current_run,
            baseline_spec=baseline,
        )
    except BaselineNotFoundError as error:
        return build_error_response(
            ApiErrorCode.BASELINE_NOT_FOUND,
            "基準実行が見つかりません。",
            {"baseline_spec": sanitize_evidence_text(error.baseline_spec)},
        )
    except BaselineInProgressError as error:
        return build_error_response(
            ApiErrorCode.BASELINE_IN_PROGRESS,
            "基準実行が実行中のため差分を作成できません。",
            {"run_id": error.run_id},
        )
    except InvalidBaselineSpecError as error:
        return build_error_response(
            ApiErrorCode.VALIDATION_ERROR,
            "基準実行の指定が不正です。",
            {
                "errors": [
                    {
                        "loc": ["query", "baseline"],
                        "msg": sanitize_evidence_text(
                            f"基準実行の指定が不正です: {error.baseline_spec}"
                        ),
                        "type": "value_error",
                    }
                ]
            },
        )

    return await build_run_diff(session, current_run, baseline_selection)


async def build_run_diff(
    session: AsyncSession,
    current_run: RunSnapshot,
    baseline_selection: BaselineSelection,
) -> JsonObject:
    baseline_run = baseline_selection.run
    baseline_results = await _load_case_results(session, baseline_run.run_id)
    current_results = await _load_case_results(session, current_run.run_id)
    current_dataset_cases = await _load_dataset_cases(session, current_run.dataset_id)
    assertions_by_case_id = await _load_assertions_by_case_id(
        session,
        case_ids=tuple(case.case_id for case in current_dataset_cases),
    )

    comparison = compare_runs(
        baseline_cases=_cases_for_comparison(baseline_results),
        current_cases=_cases_for_comparison(current_results),
    )
    classifications = classify_diff(
        baseline_cases=_case_results_for_classification(baseline_results),
        current_cases=_dataset_cases_for_classification(
            current_dataset_cases,
            current_results=current_results,
            assertions_by_case_id=assertions_by_case_id,
        ),
    )
    updated_at_warnings = collect_case_updated_at_warnings(
        baseline_run=baseline_run,
        cases=tuple(
            CaseUpdatedAt(case.case_key, case.updated_at) for case in current_dataset_cases
        ),
    )
    error_types = compare_error_type_counts(
        baseline_error_types=tuple(result.error_type for result in baseline_results),
        current_error_types=tuple(result.error_type for result in current_results),
    )

    return {
        "current_run": _run_to_json(current_run),
        "baseline_run": _run_to_json(baseline_run),
        "case_keys": _case_keys_to_json(comparison),
        "metrics": _metrics_comparison_to_json(comparison),
        "classifications": _classifications_to_json(classifications),
        "warnings": _warnings_to_json(
            baseline_selection.warnings,
            updated_at_warnings,
            baseline_run_id=baseline_run.run_id,
        ),
        "error_types": _error_type_comparison_to_json(error_types),
    }


async def _load_case_results(
    session: AsyncSession,
    run_id: int,
) -> tuple[CaseResultForDiff, ...]:
    result = await session.execute(
        select(
            CaseResultRecord.case_key,
            CaseResultRecord.status,
            CaseResultRecord.error_type,
        )
        .where(CaseResultRecord.run_id == run_id)
        .order_by(CaseResultRecord.case_key)
    )
    return tuple(
        CaseResultForDiff(
            case_key=case_key,
            status=status,
            error_type=error_type,
        )
        for case_key, status, error_type in result.all()
    )


async def _load_dataset_cases(
    session: AsyncSession,
    dataset_id: int,
) -> tuple[DatasetCaseForDiff, ...]:
    statement = text(
        """
            select case_id, case_key, is_active, updated_at
            from evaluation_cases
            where dataset_id = :dataset_id
            order by case_key
            """
    ).columns(
        case_id=Integer,
        case_key=String,
        is_active=Boolean,
        updated_at=DateTime(timezone=True),
    )
    result = await session.execute(statement, {"dataset_id": dataset_id})
    rows = result.mappings().all()
    return tuple(
        DatasetCaseForDiff(
            case_id=cast(int, row["case_id"]),
            case_key=cast(str, row["case_key"]),
            is_active=bool(row["is_active"]),
            updated_at=cast(datetime, row["updated_at"]),
        )
        for row in rows
    )


async def _load_assertions_by_case_id(
    session: AsyncSession,
    *,
    case_ids: Sequence[int],
) -> dict[int, tuple[AssertionForDiff, ...]]:
    if not case_ids:
        return {}

    result = await session.execute(
        select(AssertionRecord.case_id, AssertionRecord.id, AssertionRecord.is_active)
        .where(AssertionRecord.case_id.in_(case_ids))
        .order_by(AssertionRecord.case_id, AssertionRecord.id)
    )
    assertions_by_case_id: dict[int, list[AssertionForDiff]] = {case_id: [] for case_id in case_ids}
    for case_id, assertion_id, is_active in result.all():
        assertions_by_case_id[int(case_id)].append(
            AssertionForDiff(
                assertion_id=str(assertion_id),
                is_active=bool(is_active),
            )
        )

    return {case_id: tuple(assertions) for case_id, assertions in assertions_by_case_id.items()}


def _cases_for_comparison(
    results: Sequence[CaseResultForDiff],
) -> tuple[CaseResultForComparison, ...]:
    return tuple(CaseResultForComparison(result.case_key, result.status) for result in results)


def _case_results_for_classification(
    results: Sequence[CaseResultForDiff],
) -> tuple[CaseForDiff, ...]:
    return tuple(CaseForDiff(result.case_key, result.status) for result in results)


def _dataset_cases_for_classification(
    dataset_cases: Sequence[DatasetCaseForDiff],
    *,
    current_results: Sequence[CaseResultForDiff],
    assertions_by_case_id: dict[int, tuple[AssertionForDiff, ...]],
) -> tuple[CaseForDiff, ...]:
    status_by_case_key = {result.case_key: result.status for result in current_results}
    return tuple(
        CaseForDiff(
            case_key=case.case_key,
            status=status_by_case_key.get(case.case_key),
            is_active=case.is_active,
            assertions=assertions_by_case_id.get(case.case_id, ()),
        )
        for case in dataset_cases
    )


def _run_to_json(run: RunSnapshot) -> JsonObject:
    return {
        "run_id": run.run_id,
        "dataset_id": run.dataset_id,
        "target_label": run.target_label,
        "target_version": run.target_version,
        "status": run.status.value,
        "started_at": _datetime_to_json(run.started_at),
        "completed_at": _datetime_to_json(run.completed_at),
        "last_heartbeat_at": _datetime_to_json(run.last_heartbeat_at),
    }


def _case_keys_to_json(comparison: BaselineComparison) -> JsonObject:
    return {
        "shared": list(comparison.case_keys.shared_case_keys),
        "added": list(comparison.case_keys.added_case_keys),
        "removed": list(comparison.case_keys.removed_case_keys),
    }


def _metrics_comparison_to_json(comparison: BaselineComparison) -> JsonObject:
    return {
        "baseline": _metrics_to_json(comparison.metrics.baseline),
        "current": _metrics_to_json(comparison.metrics.current),
        "delta": {
            "total": comparison.metrics.delta.total_count,
            "executed": comparison.metrics.delta.executed_count,
            "passed": comparison.metrics.delta.passed_count,
            "failed": comparison.metrics.delta.failed_count,
            "needs_review": comparison.metrics.delta.needs_review_count,
            "skipped": comparison.metrics.delta.skipped_count,
            "pass_rate": comparison.metrics.delta.pass_rate,
            "fail_rate": comparison.metrics.delta.fail_rate,
            "skipped_ratio": comparison.metrics.delta.skipped_ratio,
        },
    }


def _metrics_to_json(metrics: RunMetrics) -> JsonObject:
    return {
        "total": metrics.total_count,
        "executed": metrics.executed_count,
        "passed": metrics.passed_count,
        "failed": metrics.failed_count,
        "needs_review": metrics.needs_review_count,
        "skipped": metrics.skipped_count,
        "pass_rate": metrics.pass_rate,
        "fail_rate": metrics.fail_rate,
        "skipped_ratio": metrics.skipped_ratio,
    }


def _classifications_to_json(
    classifications: Sequence[DiffClassification],
) -> list[JsonValue]:
    return [
        {
            "classification": classification.classification.value,
            "case_key": classification.case_key,
            "assertion_id": classification.assertion_id,
        }
        for classification in classifications
    ]


def _warnings_to_json(
    baseline_warnings: Sequence[BaselineWarning],
    updated_at_warnings: Sequence[DiffWarning],
    *,
    baseline_run_id: int,
) -> list[JsonValue]:
    warnings: list[JsonValue] = [
        {
            "code": warning.value,
            "baseline_run_id": baseline_run_id,
        }
        for warning in baseline_warnings
    ]
    warnings.extend(
        {
            "code": warning.code.value,
            "case_key": warning.case_key,
            "baseline_run_id": warning.baseline_run_id,
            "baseline_started_at": _datetime_to_json(warning.baseline_started_at),
            "case_updated_at": _datetime_to_json(warning.case_updated_at),
        }
        for warning in updated_at_warnings
    )
    return warnings


def _error_type_comparison_to_json(comparison: ErrorTypeCountComparison) -> JsonObject:
    return {
        "baseline": _error_type_counts_to_json(comparison.baseline),
        "current": _error_type_counts_to_json(comparison.current),
        "delta": _error_type_counts_to_json(comparison.delta),
    }


def _error_type_counts_to_json(error_type_counts: Mapping[str, int]) -> JsonObject:
    return {error_type: count for error_type, count in error_type_counts.items()}


def _datetime_to_json(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()
