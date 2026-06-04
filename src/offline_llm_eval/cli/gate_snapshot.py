from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from offline_llm_eval.cli.gate_config_schema import GateConfigSchema
from offline_llm_eval.cli.gate_rules import (
    GateRulesEvaluation,
    HighSeverityFailure,
    HighSeverityMustPassEvaluation,
    SecretLeakEvaluation,
    ThresholdEvaluation,
)
from offline_llm_eval.dataset.repository import JsonObject, JsonValue
from offline_llm_eval.run.heartbeat import RunRecord, utc_now
from offline_llm_eval.run.repository import RunRepository, RunSnapshot


class GateSnapshotWarning(StrEnum):
    GATE_SNAPSHOT_OVERWRITTEN = "gate_snapshot_overwritten"


@dataclass(frozen=True, slots=True)
class GateSnapshotSaveResult:
    run: RunSnapshot
    warnings: tuple[GateSnapshotWarning, ...]


async def save_gate_snapshot(
    session: AsyncSession,
    *,
    run_id: int,
    config: GateConfigSchema,
    evaluation: GateRulesEvaluation,
    evaluated_at: datetime | None = None,
) -> GateSnapshotSaveResult | None:
    result = await session.execute(select(RunRecord).where(RunRecord.run_id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        return None

    warnings = _snapshot_warnings(run)
    now = evaluated_at or utc_now()
    run.gate_config_snapshot_json = build_gate_config_snapshot(config)
    run.gate_result_json = build_gate_result_snapshot(evaluation, evaluated_at=now)
    run.updated_at = now
    await session.flush()

    snapshot = await RunRepository(session).get_run(run_id)
    if snapshot is None:
        return None
    return GateSnapshotSaveResult(run=snapshot, warnings=warnings)


def build_gate_config_snapshot(config: GateConfigSchema) -> JsonObject:
    return cast(JsonObject, config.model_dump(mode="json", exclude_none=True))


def build_gate_result_snapshot(
    evaluation: GateRulesEvaluation,
    *,
    evaluated_at: datetime,
) -> JsonObject:
    return {
        "verdict": evaluation.verdict.value,
        "evaluated_at": evaluated_at.isoformat(),
        "criteria": _criteria_to_json(evaluation),
    }


def _snapshot_warnings(run: RunRecord) -> tuple[GateSnapshotWarning, ...]:
    if run.gate_result_json is None:
        return ()
    return (GateSnapshotWarning.GATE_SNAPSHOT_OVERWRITTEN,)


def _criteria_to_json(evaluation: GateRulesEvaluation) -> list[JsonValue]:
    criteria: list[JsonValue] = []
    if evaluation.high_severity_must_pass is not None:
        criteria.append(_high_severity_to_json(evaluation.high_severity_must_pass))
    if evaluation.fail_on_secret_leak is not None:
        criteria.append(_secret_leak_to_json(evaluation.fail_on_secret_leak))
    criteria.extend(_threshold_to_json(threshold) for threshold in evaluation.thresholds)
    return criteria


def _high_severity_to_json(
    evaluation: HighSeverityMustPassEvaluation,
) -> JsonObject:
    return {
        "name": "high_severity_must_pass",
        "status": _passed_status(evaluation.passed),
        "failures": _failures_to_json(evaluation.failures),
    }


def _secret_leak_to_json(evaluation: SecretLeakEvaluation) -> JsonObject:
    return {
        "name": "fail_on_secret_leak",
        "status": _passed_status(evaluation.passed),
        "failures": _failures_to_json(evaluation.failures),
    }


def _threshold_to_json(evaluation: ThresholdEvaluation) -> JsonObject:
    return {
        "name": evaluation.criterion.value,
        "status": evaluation.status.value,
        "actual": evaluation.actual,
        "threshold": evaluation.threshold,
        "reason": evaluation.reason,
    }


def _failures_to_json(failures: tuple[HighSeverityFailure, ...]) -> list[JsonValue]:
    return [
        {
            "case_key": failure.case_key,
            "assertion_id": failure.assertion_id,
            "reason": failure.reason.value,
        }
        for failure in failures
    ]


def _passed_status(passed: bool) -> str:
    if passed:
        return "passed"
    return "failed"
