from collections.abc import Sequence
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from offline_llm_eval.api.error_response import ApiErrorCode, build_error_response
from offline_llm_eval.api.runs import get_database_session
from offline_llm_eval.dataset.repository import JsonObject, JsonValue
from offline_llm_eval.evaluator.results_schema import (
    NormalAssertionResultDbSchema,
    PseudoEvaluatorResultDbSchema,
    dump_unified_evaluator_result,
    merge_evaluator_results,
    validate_normal_assertion_result,
    validate_pseudo_evaluator_result,
)
from offline_llm_eval.evidence.sanitize import (
    sanitize_evidence_text,
    sanitize_evidence_value,
    sanitize_optional_reviewer_verdict,
    sanitize_reviewer_note,
)
from offline_llm_eval.run.case_result import AssertionResultRecord, CaseResultRecord
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus

router = APIRouter(prefix="/api/runs/{run_id}/cases")


@router.get("/{case_key}", response_model=None)
async def get_case_detail(
    run_id: int,
    case_key: str,
    session: Annotated[AsyncSession, Depends(get_database_session)],
) -> JsonObject | JSONResponse:
    detail = await build_case_detail(session, run_id=run_id, case_key=case_key)
    if detail is not None:
        return detail

    if not await _run_exists(session, run_id):
        return build_error_response(
            ApiErrorCode.RUN_NOT_FOUND,
            "実行が見つかりません。",
            {"run_id": run_id},
        )

    return build_error_response(
        ApiErrorCode.CASE_NOT_FOUND,
        "ケースが見つかりません。",
        {"run_id": run_id, "case_key": case_key},
    )


async def build_case_detail(
    session: AsyncSession,
    *,
    run_id: int,
    case_key: str,
) -> JsonObject | None:
    run_and_case = await _load_run_and_case(session, run_id=run_id, case_key=case_key)
    if run_and_case is None:
        return None

    run, case = run_and_case
    normal_results = await _load_normal_assertion_results(session, case.case_result_id)
    pseudo_results = _load_pseudo_results(case.evaluator_results_json)
    assertions = _dump_merged_assertions(normal_results, pseudo_results)

    return {
        "run": {
            "run_id": run.run_id,
            "target_label": run.target_label,
            "target_version": run.target_version,
            "status": RunStatus(run.status).value,
        },
        "case": {
            "case_result_id": case.case_result_id,
            "case_id": case.case_id,
            "case_key": case.case_key,
            "status": case.status,
            "error_type": case.error_type,
        },
        "review": {
            "reviewer_verdict": sanitize_optional_reviewer_verdict(case.reviewer_verdict),
            "reviewer_note": sanitize_reviewer_note(case.reviewer_note),
            "reviewed_at": _datetime_to_json(case.reviewed_at),
            "final_status": case.final_status,
        },
        "assertions": assertions,
        "failed_assertions": _failed_assertions(assertions),
    }


async def _load_run_and_case(
    session: AsyncSession,
    *,
    run_id: int,
    case_key: str,
) -> tuple[RunRecord, CaseResultRecord] | None:
    result = await session.execute(
        select(RunRecord, CaseResultRecord)
        .join(CaseResultRecord, CaseResultRecord.run_id == RunRecord.run_id)
        .where(
            RunRecord.run_id == run_id,
            CaseResultRecord.case_key == case_key,
        )
    )
    row = result.one_or_none()
    if row is None:
        return None

    return row[0], row[1]


async def _run_exists(session: AsyncSession, run_id: int) -> bool:
    result = await session.execute(select(RunRecord.run_id).where(RunRecord.run_id == run_id))
    return result.scalar_one_or_none() is not None


async def _load_normal_assertion_results(
    session: AsyncSession,
    case_result_id: int,
) -> tuple[NormalAssertionResultDbSchema, ...]:
    result = await session.execute(
        select(AssertionResultRecord)
        .where(AssertionResultRecord.case_result_id == case_result_id)
        .order_by(AssertionResultRecord.assertion_result_id)
    )
    records = tuple(result.scalars().all())
    return tuple(
        validate_normal_assertion_result(_normal_result_payload(record)) for record in records
    )


def _normal_result_payload(record: AssertionResultRecord) -> JsonObject:
    return {
        "assertion_db_id": record.assertion_db_id,
        "assertion_id": record.assertion_id,
        "assertion_type": record.assertion_type,
        "status": record.status,
        "detail": _sanitize_detail(record.detail),
        "matched_value": sanitize_evidence_value(record.matched_value_json),
        "expected": sanitize_evidence_value(record.expected_json),
        "required": record.required,
        "severity": record.severity,
        "on_fail": record.on_fail,
    }


def _sanitize_detail(detail: str | None) -> str | None:
    if detail is None:
        return None
    return sanitize_evidence_text(detail)


def _load_pseudo_results(value: JsonValue | None) -> tuple[PseudoEvaluatorResultDbSchema, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        return ()

    return tuple(validate_pseudo_evaluator_result(item) for item in value)


def _dump_merged_assertions(
    normal_results: tuple[NormalAssertionResultDbSchema, ...],
    pseudo_results: tuple[PseudoEvaluatorResultDbSchema, ...],
) -> list[JsonValue]:
    return [
        dump_unified_evaluator_result(result)
        for result in merge_evaluator_results(normal_results, pseudo_results)
    ]


def _failed_assertions(assertions: Sequence[JsonValue]) -> list[JsonValue]:
    return [
        assertion
        for assertion in assertions
        if isinstance(assertion, dict) and assertion.get("status") == "failed"
    ]


def _datetime_to_json(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()
