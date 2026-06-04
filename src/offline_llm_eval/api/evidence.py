from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from offline_llm_eval.api.error_response import ApiErrorCode, build_error_response
from offline_llm_eval.api.runs import get_database_session
from offline_llm_eval.dataset.repository import Dataset, JsonObject, JsonValue
from offline_llm_eval.evaluator.results_schema import (
    NormalAssertionResultDbSchema,
    PseudoEvaluatorResultDbSchema,
    UnifiedEvaluatorResultSchema,
    merge_evaluator_results,
    validate_normal_assertion_result,
    validate_pseudo_evaluator_result,
)
from offline_llm_eval.evidence.markdown import (
    EvidenceAssertion,
    EvidenceCase,
    EvidenceReport,
    EvidenceRunMetadata,
    render_evidence_markdown,
)
from offline_llm_eval.evidence.sanitize import (
    sanitize_evidence_text,
    sanitize_evidence_value,
    sanitize_reviewer_note,
)
from offline_llm_eval.run.case_result import AssertionResultRecord, CaseResultRecord
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus
from offline_llm_eval.run.metrics import calculate_run_metrics, resolve_effective_case_status

router = APIRouter(prefix="/api/runs")


@router.get("/{run_id}/evidence", response_model=None)
async def get_run_evidence(
    run_id: int,
    session: Annotated[AsyncSession, Depends(get_database_session)],
) -> PlainTextResponse | JSONResponse:
    report = await build_evidence_report(session, run_id)
    if report is None:
        return build_error_response(
            ApiErrorCode.RUN_NOT_FOUND,
            "実行が見つかりません。",
            {"run_id": run_id},
        )
    return PlainTextResponse(
        render_evidence_markdown(report),
        media_type="text/markdown",
    )


async def build_evidence_report(session: AsyncSession, run_id: int) -> EvidenceReport | None:
    run_and_dataset = await _load_run_and_dataset(session, run_id)
    if run_and_dataset is None:
        return None

    run, dataset = run_and_dataset
    cases = await _load_evidence_cases(session, run_id=run_id)
    metrics = calculate_run_metrics(
        tuple(resolve_effective_case_status(case.status, case.final_status) for case in cases)
    )
    return EvidenceReport(
        run=EvidenceRunMetadata(
            run_id=run.run_id,
            dataset_name=dataset.name,
            dataset_version=dataset.dataset_version,
            target_label=run.target_label,
            target_version=run.target_version,
            status=RunStatus(run.status).value,
            started_at=run.started_at,
            completed_at=run.completed_at,
        ),
        metrics=metrics,
        cases=cases,
    )


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


async def _load_evidence_cases(
    session: AsyncSession,
    *,
    run_id: int,
) -> tuple[EvidenceCase, ...]:
    result = await session.execute(
        select(CaseResultRecord)
        .where(CaseResultRecord.run_id == run_id)
        .order_by(CaseResultRecord.case_key)
    )
    cases = tuple(result.scalars().all())
    evidence_cases: list[EvidenceCase] = []
    for case in cases:
        evidence_cases.append(
            EvidenceCase(
                case_key=case.case_key,
                status=case.status,
                final_status=case.final_status,
                reviewer_note=sanitize_reviewer_note(case.reviewer_note),
                assertions=await _load_evidence_assertions(session, case),
            )
        )
    return tuple(evidence_cases)


async def _load_evidence_assertions(
    session: AsyncSession,
    case: CaseResultRecord,
) -> tuple[EvidenceAssertion, ...]:
    normal_results = await _load_normal_assertion_results(session, case.case_result_id)
    pseudo_results = _load_pseudo_results(case.evaluator_results_json)
    return tuple(
        _evidence_assertion(result)
        for result in merge_evaluator_results(normal_results, pseudo_results)
    )


async def _load_normal_assertion_results(
    session: AsyncSession,
    case_result_id: int,
) -> tuple[NormalAssertionResultDbSchema, ...]:
    result = await session.execute(
        select(AssertionResultRecord)
        .where(AssertionResultRecord.case_result_id == case_result_id)
        .order_by(AssertionResultRecord.assertion_result_id)
    )
    return tuple(
        validate_normal_assertion_result(_normal_result_payload(record))
        for record in result.scalars().all()
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


def _evidence_assertion(result: UnifiedEvaluatorResultSchema) -> EvidenceAssertion:
    return EvidenceAssertion(
        assertion_id=result.assertion_id,
        assertion_type=result.assertion_type.value,
        status=result.status.value,
        detail=result.detail,
        required=result.required,
        severity=result.severity.value,
        on_fail=result.on_fail.value,
    )
