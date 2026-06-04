from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from offline_llm_eval.evaluator.secret_pattern import SECRET_SCAN_ASSERTION_ID
from offline_llm_eval.evidence.sanitize import sanitize_evidence_text, sanitize_reviewer_note
from offline_llm_eval.run.metrics import RunMetrics

DEFAULT_NOT_CLAIMED: Final = (
    "運用環境でそのまま使えること",
    "あらゆるLLM出力品質を評価できること",
    "外部LLMサービスの費用を制御できること",
    "長時間運用に耐えること",
)


@dataclass(frozen=True, slots=True)
class EvidenceRunMetadata:
    run_id: int
    dataset_name: str
    dataset_version: str
    target_label: str
    target_version: str | None
    status: str
    started_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class EvidenceAssertion:
    assertion_id: str
    assertion_type: str
    status: str
    detail: str | None
    required: bool
    severity: str
    on_fail: str


@dataclass(frozen=True, slots=True)
class EvidenceCase:
    case_key: str
    status: str
    final_status: str | None
    reviewer_note: str | None
    assertions: tuple[EvidenceAssertion, ...]


@dataclass(frozen=True, slots=True)
class EvidenceReport:
    run: EvidenceRunMetadata
    metrics: RunMetrics
    cases: tuple[EvidenceCase, ...]
    not_claimed: tuple[str, ...] = DEFAULT_NOT_CLAIMED


def render_evidence_markdown(report: EvidenceReport) -> str:
    sections = [
        "# 実行証跡",
        _render_run_metadata(report.run),
        _render_case_summary(report.metrics),
        _render_failed_cases(report.cases),
        _render_warning_cases(report.cases),
        _render_secret_leak(report.cases),
        _render_not_claimed(report.not_claimed),
    ]
    return "\n\n".join(sections) + "\n"


def _render_run_metadata(run: EvidenceRunMetadata) -> str:
    return "\n".join(
        [
            "## 実行情報（run metadata）",
            f"- run_id: {run.run_id}",
            f"- dataset: {_inline(run.dataset_name)}@{_inline(run.dataset_version)}",
            f"- target_label: {_inline(run.target_label)}",
            f"- target_version: {_inline(run.target_version) if run.target_version else 'null'}",
            f"- status: {_inline(run.status)}",
            f"- started_at: {run.started_at.isoformat()}",
            f"- completed_at: {_datetime_to_text(run.completed_at)}",
        ]
    )


def _render_case_summary(metrics: RunMetrics) -> str:
    return "\n".join(
        [
            "## ケース集計（case summary）",
            f"- total: {metrics.total_count}",
            f"- executed: {metrics.executed_count}",
            f"- passed: {metrics.passed_count}",
            f"- failed: {metrics.failed_count}",
            f"- needs_review: {metrics.needs_review_count}",
            f"- skipped: {metrics.skipped_count}",
            f"- pass_rate: {metrics.pass_rate:.4f}",
            f"- fail_rate: {metrics.fail_rate:.4f}",
            f"- skipped_ratio: {metrics.skipped_ratio:.4f}",
        ]
    )


def _render_failed_cases(cases: Sequence[EvidenceCase]) -> str:
    lines = ["## 失敗ケース（failed cases）"]
    failed_cases = tuple(case for case in cases if _blocking_failures(case))
    if not failed_cases:
        lines.append("- none")
        return "\n".join(lines)

    for case in failed_cases:
        lines.append(f"### {_inline(case.case_key)}")
        lines.append(f"- status: {_inline(case.status)}")
        lines.append(
            f"- final_status: {_inline(case.final_status) if case.final_status else 'null'}"
        )
        lines.append(f"- reviewer_note: {_reviewer_note_to_text(case.reviewer_note)}")
        _append_assertions(lines, _blocking_failures(case))
    return "\n".join(lines)


def _render_warning_cases(cases: Sequence[EvidenceCase]) -> str:
    lines = ["## 注意ケース（warning cases）"]
    warning_cases = tuple(case for case in cases if _warning_assertions(case))
    if not warning_cases:
        lines.append("- none")
        return "\n".join(lines)

    for case in warning_cases:
        lines.append(f"### {_inline(case.case_key)}")
        lines.append(f"- reviewer_note: {_reviewer_note_to_text(case.reviewer_note)}")
        _append_assertions(lines, _warning_assertions(case))
    return "\n".join(lines)


def _render_secret_leak(cases: Sequence[EvidenceCase]) -> str:
    lines = ["## secret検出（secret leak）"]
    secret_assertions = tuple(
        (case, assertion)
        for case in cases
        for assertion in case.assertions
        if _is_failed_secret_scan(assertion)
    )
    if not secret_assertions:
        lines.append("- none")
        return "\n".join(lines)

    for case, assertion in secret_assertions:
        detail = _inline(assertion.detail) if assertion.detail is not None else "null"
        lines.append(f"- {case.case_key}: {assertion.assertion_id} ({detail})")
    return "\n".join(lines)


def _render_not_claimed(not_claimed: Sequence[str]) -> str:
    lines = ["## 主張しない範囲（not_claimed）"]
    if not not_claimed:
        lines.append("- none")
        return "\n".join(lines)

    lines.extend(f"- {_inline(item)}" for item in not_claimed)
    return "\n".join(lines)


def _append_assertions(lines: list[str], assertions: Sequence[EvidenceAssertion]) -> None:
    for assertion in assertions:
        detail = _inline(assertion.detail) if assertion.detail is not None else "null"
        lines.append(
            "- "
            f"{_inline(assertion.assertion_id)} "
            f"({assertion.assertion_type}, status={assertion.status}, "
            f"severity={assertion.severity}, required={assertion.required}, "
            f"on_fail={assertion.on_fail}, detail={detail})"
        )


def _blocking_failures(case: EvidenceCase) -> tuple[EvidenceAssertion, ...]:
    return tuple(
        assertion
        for assertion in case.assertions
        if assertion.status == "failed"
        and not _is_warning_assertion(assertion)
        and not _is_failed_secret_scan(assertion)
    )


def _warning_assertions(case: EvidenceCase) -> tuple[EvidenceAssertion, ...]:
    return tuple(assertion for assertion in case.assertions if _is_warning_assertion(assertion))


def _is_warning_assertion(assertion: EvidenceAssertion) -> bool:
    if assertion.status == "warning":
        return True
    return assertion.status == "failed" and (not assertion.required or assertion.on_fail == "warn")


def _is_failed_secret_scan(assertion: EvidenceAssertion) -> bool:
    return assertion.assertion_id == SECRET_SCAN_ASSERTION_ID and assertion.status == "failed"


def _inline(value: str | None) -> str:
    if value is None:
        return "null"
    return sanitize_evidence_text(value).replace("\r", " ").replace("\n", " ")


def _reviewer_note_to_text(value: str | None) -> str:
    sanitized = sanitize_reviewer_note(value)
    if sanitized is None:
        return "null"
    return sanitized.replace("\r", " ").replace("\n", " ")


def _datetime_to_text(value: datetime | None) -> str:
    if value is None:
        return "null"
    return value.isoformat()
