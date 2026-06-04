import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO, cast

from pydantic import ValidationError
from sqlalchemy import JSON, Integer, String, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from offline_llm_eval.cli.gate_config import load_gate_config
from offline_llm_eval.cli.gate_config_schema import GateConfigSchema
from offline_llm_eval.cli.gate_rules import (
    GateAssertionResult,
    GateCaseResult,
    GateRulesEvaluation,
    GateRulesVerdict,
    evaluate_gate_rules,
)
from offline_llm_eval.cli.gate_snapshot import GateSnapshotWarning, save_gate_snapshot
from offline_llm_eval.dataset.assertion_model import AssertionOnFail, AssertionSeverity
from offline_llm_eval.dataset.repository import JsonValue
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.diff.baseline_selector import (
    RUN_SPEC_PREFIX,
    BaselineInProgressError,
    BaselineNotFoundError,
    InvalidBaselineSpecError,
    select_baseline_run,
)
from offline_llm_eval.diff.comparator import CaseResultForComparison, RunMetricsDelta, compare_runs
from offline_llm_eval.evaluator.assertions import AssertionEvaluationStatus
from offline_llm_eval.evaluator.results_schema import (
    InvalidEvaluatorResultSchemaError,
    validate_pseudo_evaluator_result,
)
from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.run.case_result import AssertionResultRecord
from offline_llm_eval.run.metrics import calculate_run_metrics
from offline_llm_eval.run.repository import RunRepository, RunSnapshot
from offline_llm_eval.util.document import DocumentLoadError

GATE_PASSED_EXIT_CODE = 0
GATE_FAILED_EXIT_CODE = 1
GATE_USAGE_ERROR_EXIT_CODE = 2
DEFAULT_BASELINE_SPEC = "latest"


@dataclass(frozen=True, slots=True)
class GateCliOptions:
    run_id: int
    config_path: Path | None = None
    baseline_spec: str | None = DEFAULT_BASELINE_SPEC
    pass_rate_min: float | None = None
    fail_rate_max: float | None = None
    max_pass_rate_delta: float | None = None
    max_fail_rate_delta: float | None = None
    max_skipped_ratio_delta: float | None = None
    high_severity_must_pass: bool | None = None
    fail_on_secret_leak: bool | None = None
    max_high_severity_skipped: int | None = None


@dataclass(frozen=True, slots=True)
class GateCliResult:
    exit_code: int
    evaluation: GateRulesEvaluation | None
    run: RunSnapshot | None
    warnings: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class GateCaseRow:
    case_result_id: int
    case_key: str
    status: str
    final_status: str | None
    evaluator_results_json: JsonValue | None
    case_severity: str


class GateCliError(ValueError):
    code = "gate_cli_error"
    exit_code = GATE_USAGE_ERROR_EXIT_CODE


class RunNotFoundError(GateCliError):
    code = "run_not_found"
    exit_code = GATE_USAGE_ERROR_EXIT_CODE

    def __init__(self, run_id: int) -> None:
        self.run_id = run_id
        super().__init__(f"{self.code}: 実行が見つかりません: run_id={run_id}")


class GateDataValidationError(GateCliError):
    code = "validation_error"
    exit_code = GATE_USAGE_ERROR_EXIT_CODE

    def __init__(self, message: str) -> None:
        super().__init__(f"{self.code}: {message}")


async def run_gate_from_database(
    options: GateCliOptions,
    *,
    evaluated_at: datetime | None = None,
) -> GateCliResult:
    engine = create_async_db_engine()
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            return await run_gate(
                session,
                options=options,
                evaluated_at=evaluated_at,
            )
    finally:
        await engine.dispose()


async def run_gate(
    session: AsyncSession,
    *,
    options: GateCliOptions,
    evaluated_at: datetime | None = None,
) -> GateCliResult:
    try:
        config = load_gate_config(
            config_path=options.config_path,
            cli_overrides=_cli_overrides(options),
        )
    except (DocumentLoadError, OSError, ValidationError) as error:
        return _error_result(GATE_USAGE_ERROR_EXIT_CODE, _config_error_message(error))

    try:
        return await _evaluate_and_save_gate(
            session,
            run_id=options.run_id,
            config=config,
            baseline_spec=options.baseline_spec,
            evaluated_at=evaluated_at,
        )
    except GateCliError as error:
        return _error_result(error.exit_code, str(error))
    except BaselineInProgressError as error:
        return _error_result(error.exit_code, str(error))
    except BaselineNotFoundError as error:
        return _error_result(GATE_USAGE_ERROR_EXIT_CODE, str(error))
    except InvalidBaselineSpecError as error:
        return _error_result(GATE_USAGE_ERROR_EXIT_CODE, str(error))
    except InvalidEvaluatorResultSchemaError as error:
        return _error_result(GATE_USAGE_ERROR_EXIT_CODE, str(error))


def main(argv: Sequence[str] | None = None, stderr: TextIO | None = None) -> int:
    error_output = stderr
    try:
        options = parse_args(argv)
    except SystemExit as error:
        return _system_exit_code(error.code)

    result = asyncio.run(run_gate_from_database(options))
    _write_stderr(result, error_output)
    return result.exit_code


def parse_args(argv: Sequence[str] | None = None) -> GateCliOptions:
    parser = argparse.ArgumentParser(prog="offline-llm-eval-check")
    parser.add_argument("--run", dest="run_id", type=int, required=True)
    parser.add_argument("--config", dest="config_path", type=Path)
    parser.add_argument("--baseline", dest="baseline_spec", default=DEFAULT_BASELINE_SPEC)
    parser.add_argument("--pass-rate-min", type=float)
    parser.add_argument("--fail-rate-max", type=float)
    parser.add_argument("--max-pass-rate-delta", type=float)
    parser.add_argument("--max-fail-rate-delta", type=float)
    parser.add_argument("--max-skipped-ratio-delta", type=float)
    high_severity_group = parser.add_mutually_exclusive_group()
    high_severity_group.add_argument(
        "--high-severity-must-pass",
        dest="high_severity_must_pass",
        action="store_const",
        const=True,
        default=None,
    )
    high_severity_group.add_argument(
        "--no-high-severity-must-pass",
        dest="high_severity_must_pass",
        action="store_const",
        const=False,
    )
    secret_leak_group = parser.add_mutually_exclusive_group()
    secret_leak_group.add_argument(
        "--fail-on-secret-leak",
        dest="fail_on_secret_leak",
        action="store_const",
        const=True,
        default=None,
    )
    secret_leak_group.add_argument(
        "--no-fail-on-secret-leak",
        dest="fail_on_secret_leak",
        action="store_const",
        const=False,
    )
    parser.add_argument("--max-high-severity-skipped", type=int)
    namespace = parser.parse_args(argv)
    return GateCliOptions(
        run_id=cast(int, namespace.run_id),
        config_path=cast(Path | None, namespace.config_path),
        baseline_spec=cast(str | None, namespace.baseline_spec),
        pass_rate_min=cast(float | None, namespace.pass_rate_min),
        fail_rate_max=cast(float | None, namespace.fail_rate_max),
        max_pass_rate_delta=cast(float | None, namespace.max_pass_rate_delta),
        max_fail_rate_delta=cast(float | None, namespace.max_fail_rate_delta),
        max_skipped_ratio_delta=cast(float | None, namespace.max_skipped_ratio_delta),
        high_severity_must_pass=cast(bool | None, namespace.high_severity_must_pass),
        fail_on_secret_leak=cast(bool | None, namespace.fail_on_secret_leak),
        max_high_severity_skipped=cast(int | None, namespace.max_high_severity_skipped),
    )


async def _evaluate_and_save_gate(
    session: AsyncSession,
    *,
    run_id: int,
    config: GateConfigSchema,
    baseline_spec: str | None,
    evaluated_at: datetime | None,
) -> GateCliResult:
    current_run = await RunRepository(session).get_run(run_id)
    if current_run is None:
        raise RunNotFoundError(run_id)

    current_cases = await _load_gate_cases(session, run_id=current_run.run_id)
    current_metrics = calculate_run_metrics(_case_statuses(current_cases))
    metrics_delta = await _resolve_metrics_delta(
        session,
        current_run=current_run,
        current_cases=current_cases,
        baseline_spec=baseline_spec,
    )
    evaluation = evaluate_gate_rules(
        config,
        current_metrics=current_metrics,
        metrics_delta=metrics_delta,
        cases=current_cases,
    )
    save_result = await save_gate_snapshot(
        session,
        run_id=current_run.run_id,
        config=config,
        evaluation=evaluation,
        evaluated_at=evaluated_at,
    )
    if save_result is None:
        raise RunNotFoundError(run_id)

    return GateCliResult(
        exit_code=_exit_code_for_evaluation(evaluation),
        evaluation=evaluation,
        run=save_result.run,
        warnings=_snapshot_warning_values(save_result.warnings),
    )


async def _resolve_metrics_delta(
    session: AsyncSession,
    *,
    current_run: RunSnapshot,
    current_cases: Sequence[GateCaseResult],
    baseline_spec: str | None,
) -> RunMetricsDelta | None:
    if baseline_spec is None:
        return None

    try:
        baseline_selection = await select_baseline_run(
            session,
            current_run=current_run,
            baseline_spec=baseline_spec,
        )
    except BaselineNotFoundError as error:
        if baseline_spec.startswith(RUN_SPEC_PREFIX):
            raise error
        return None

    baseline_cases = await _load_cases_for_comparison(
        session,
        run_id=baseline_selection.run.run_id,
    )
    comparison = compare_runs(
        baseline_cases=baseline_cases,
        current_cases=_cases_for_comparison(current_cases),
    )
    return comparison.metrics.delta


async def _load_gate_cases(
    session: AsyncSession,
    *,
    run_id: int,
) -> tuple[GateCaseResult, ...]:
    case_rows = await _load_gate_case_rows(session, run_id=run_id)
    assertions_by_case_id = await _load_assertions_by_case_result_id(
        session,
        case_result_ids=tuple(row.case_result_id for row in case_rows),
    )
    return tuple(
        GateCaseResult(
            case_key=row.case_key,
            assertions=(
                *assertions_by_case_id.get(row.case_result_id, ()),
                *_pseudo_assertions(row.evaluator_results_json),
            ),
            status=CaseResultStatus(row.status),
            case_severity=AssertionSeverity(row.case_severity),
            final_status=row.final_status,
        )
        for row in case_rows
    )


async def _load_gate_case_rows(
    session: AsyncSession,
    *,
    run_id: int,
) -> tuple[GateCaseRow, ...]:
    statement = text(
        """
        select
            cr.case_result_id,
            cr.case_key,
            cr.status,
            cr.final_status,
            cr.evaluator_results_json,
            ec.severity as case_severity
        from case_results cr
        join evaluation_cases ec on ec.case_id = cr.case_id
        where cr.run_id = :run_id
        order by cr.case_key
        """
    ).columns(
        case_result_id=Integer,
        case_key=String,
        status=String,
        final_status=String,
        evaluator_results_json=JSON,
        case_severity=String,
    )
    result = await session.execute(statement, {"run_id": run_id})
    return tuple(
        GateCaseRow(
            case_result_id=cast(int, row["case_result_id"]),
            case_key=cast(str, row["case_key"]),
            status=cast(str, row["status"]),
            final_status=cast(str | None, row["final_status"]),
            evaluator_results_json=cast(JsonValue | None, row["evaluator_results_json"]),
            case_severity=cast(str, row["case_severity"]),
        )
        for row in result.mappings().all()
    )


async def _load_assertions_by_case_result_id(
    session: AsyncSession,
    *,
    case_result_ids: Sequence[int],
) -> dict[int, tuple[GateAssertionResult, ...]]:
    if not case_result_ids:
        return {}

    result = await session.execute(
        select(
            AssertionResultRecord.case_result_id,
            AssertionResultRecord.assertion_id,
            AssertionResultRecord.status,
            AssertionResultRecord.required,
            AssertionResultRecord.severity,
            AssertionResultRecord.on_fail,
        )
        .where(AssertionResultRecord.case_result_id.in_(case_result_ids))
        .order_by(
            AssertionResultRecord.case_result_id,
            AssertionResultRecord.assertion_result_id,
        )
    )
    assertions_by_case_result_id: dict[int, list[GateAssertionResult]] = {
        case_result_id: [] for case_result_id in case_result_ids
    }
    for case_result_id, assertion_id, status, required, severity, on_fail in result.all():
        assertions_by_case_result_id[int(case_result_id)].append(
            GateAssertionResult(
                assertion_id=str(assertion_id),
                status=AssertionEvaluationStatus(str(status)),
                required=bool(required),
                severity=AssertionSeverity(str(severity)),
                on_fail=AssertionOnFail(str(on_fail)),
            )
        )
    return {
        case_result_id: tuple(assertions)
        for case_result_id, assertions in assertions_by_case_result_id.items()
    }


async def _load_cases_for_comparison(
    session: AsyncSession,
    *,
    run_id: int,
) -> tuple[CaseResultForComparison, ...]:
    result = await session.execute(
        text(
            """
            select case_key, status
            from case_results
            where run_id = :run_id
            order by case_key
            """
        ).columns(case_key=String, status=String),
        {"run_id": run_id},
    )
    return tuple(
        CaseResultForComparison(
            case_key=cast(str, row["case_key"]),
            status=cast(str, row["status"]),
        )
        for row in result.mappings().all()
    )


def _pseudo_assertions(value: JsonValue | None) -> tuple[GateAssertionResult, ...]:
    if value is None:
        return ()

    if not isinstance(value, list):
        raise GateDataValidationError("evaluator_results_json はlistである必要があります。")

    return tuple(_pseudo_assertion(item) for item in value)


def _pseudo_assertion(value: JsonValue) -> GateAssertionResult:
    result = validate_pseudo_evaluator_result(value)
    return GateAssertionResult(
        assertion_id=result.id,
        status=AssertionEvaluationStatus(result.status.value),
        required=result.required,
        severity=result.severity,
        on_fail=result.on_fail,
    )


def _cases_for_comparison(
    cases: Sequence[GateCaseResult],
) -> tuple[CaseResultForComparison, ...]:
    return tuple(
        CaseResultForComparison(case_key=case.case_key, status=case.status.value)
        for case in cases
        if case.status is not None
    )


def _case_statuses(cases: Sequence[GateCaseResult]) -> tuple[str, ...]:
    return tuple(case.status.value for case in cases if case.status is not None)


def _cli_overrides(options: GateCliOptions) -> dict[str, object]:
    return {
        "pass_rate_min": options.pass_rate_min,
        "fail_rate_max": options.fail_rate_max,
        "max_pass_rate_delta": options.max_pass_rate_delta,
        "max_fail_rate_delta": options.max_fail_rate_delta,
        "max_skipped_ratio_delta": options.max_skipped_ratio_delta,
        "high_severity_must_pass": options.high_severity_must_pass,
        "fail_on_secret_leak": options.fail_on_secret_leak,
        "max_high_severity_skipped": options.max_high_severity_skipped,
    }


def _exit_code_for_evaluation(evaluation: GateRulesEvaluation) -> int:
    if evaluation.verdict is GateRulesVerdict.PASSED:
        return GATE_PASSED_EXIT_CODE
    return GATE_FAILED_EXIT_CODE


def _snapshot_warning_values(
    warnings: Sequence[GateSnapshotWarning],
) -> tuple[str, ...]:
    return tuple(warning.value for warning in warnings)


def _error_result(exit_code: int, message: str) -> GateCliResult:
    return GateCliResult(
        exit_code=exit_code,
        evaluation=None,
        run=None,
        error=message,
    )


def _config_error_message(error: DocumentLoadError | OSError | ValidationError) -> str:
    if isinstance(error, DocumentLoadError):
        return str(error)

    if isinstance(error, OSError):
        return f"validation_error: 品質判定設定を読み込めません: {error}"

    first_error = error.errors()[0]
    loc = ".".join(str(part) for part in first_error["loc"])
    return f"validation_error: {loc}: 入力値が不正です: {first_error['msg']}"


def _write_stderr(result: GateCliResult, stderr: TextIO | None) -> None:
    if stderr is None:
        import sys

        stderr = sys.stderr

    for warning in result.warnings:
        stderr.write(f"{warning}\n")
    if result.error is not None:
        stderr.write(f"{result.error}\n")


def _system_exit_code(code: str | int | None) -> int:
    if isinstance(code, int):
        return code
    return GATE_USAGE_ERROR_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
