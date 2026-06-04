from datetime import datetime

from offline_llm_eval.evaluator.secret_pattern import SECRET_SCAN_ASSERTION_ID
from offline_llm_eval.evidence.markdown import (
    EvidenceAssertion,
    EvidenceCase,
    EvidenceReport,
    EvidenceRunMetadata,
    render_evidence_markdown,
)
from offline_llm_eval.run.metrics import calculate_run_metrics

STARTED_AT = datetime(2026, 5, 27, 15, 0, 0)
COMPLETED_AT = datetime(2026, 5, 27, 15, 5, 0)


def test_evidence_markdownは主要セクションを出力する() -> None:
    report = EvidenceReport(
        run=EvidenceRunMetadata(
            run_id=7,
            dataset_name="sample_dataset",
            dataset_version="1.0.0",
            target_label="local",
            target_version="mock-v1",
            status="completed",
            started_at=STARTED_AT,
            completed_at=COMPLETED_AT,
        ),
        metrics=calculate_run_metrics(("pass", "failed", "needs_review", "skipped")),
        cases=(
            EvidenceCase(
                case_key="case_failed",
                status="failed",
                final_status=None,
                reviewer_note=None,
                assertions=(
                    EvidenceAssertion(
                        assertion_id="answer_exact",
                        assertion_type="exact_match",
                        status="failed",
                        detail="exact_match_mismatch",
                        required=True,
                        severity="high",
                        on_fail="fail",
                    ),
                ),
            ),
        ),
        not_claimed=("運用環境でそのまま使えること", "外部LLMサービスの費用を制御できること"),
    )

    markdown = render_evidence_markdown(report)

    assert "# 実行証跡" in markdown
    assert "## 実行情報（run metadata）" in markdown
    assert "- run_id: 7" in markdown
    assert "- dataset: sample_dataset@1.0.0" in markdown
    assert "## ケース集計（case summary）" in markdown
    assert "- total: 4" in markdown
    assert "- pass_rate: 0.3333" in markdown
    assert "## 失敗ケース（failed cases）" in markdown
    assert "### case_failed" in markdown
    assert "answer_exact (exact_match, status=failed" in markdown
    assert "## secret検出（secret leak）\n- none" in markdown
    assert "## 主張しない範囲（not_claimed）" in markdown
    assert "- 外部LLMサービスの費用を制御できること" in markdown


def test_evidence_markdownはsecret_leakを専用sectionに出す() -> None:
    report = EvidenceReport(
        run=run_metadata(),
        metrics=calculate_run_metrics(("failed",)),
        cases=(
            EvidenceCase(
                case_key="case_secret",
                status="failed",
                final_status="pass",
                reviewer_note="approved after review",
                assertions=(
                    EvidenceAssertion(
                        assertion_id=SECRET_SCAN_ASSERTION_ID,
                        assertion_type="secret_scan",
                        status="failed",
                        detail="aws_access_key",
                        required=True,
                        severity="high",
                        on_fail="fail",
                    ),
                ),
            ),
        ),
    )

    markdown = render_evidence_markdown(report)

    assert "## 失敗ケース（failed cases）\n- none" in markdown
    assert "## secret検出（secret leak）" in markdown
    assert "- case_secret: __secret_scan__ (aws_access_key)" in markdown


def test_evidence_markdownはrequired_false_failedをwarning扱いにする() -> None:
    report = EvidenceReport(
        run=run_metadata(),
        metrics=calculate_run_metrics(("pass",)),
        cases=(
            EvidenceCase(
                case_key="case_warning",
                status="pass",
                final_status=None,
                reviewer_note=None,
                assertions=(
                    EvidenceAssertion(
                        assertion_id="optional_keyword",
                        assertion_type="keyword_any",
                        status="failed",
                        detail="missing_optional_keyword",
                        required=False,
                        severity="medium",
                        on_fail="warn",
                    ),
                ),
            ),
        ),
    )

    markdown = render_evidence_markdown(report)

    assert "## 失敗ケース（failed cases）\n- none" in markdown
    assert "## 注意ケース（warning cases）" in markdown
    assert "### case_warning" in markdown
    assert "optional_keyword (keyword_any, status=failed" in markdown


def test_evidence_markdownはsecret値をmaskする() -> None:
    secret_value = "".join(("AKIA", "12345678", "90ABCDEF"))
    report = EvidenceReport(
        run=EvidenceRunMetadata(
            run_id=8,
            dataset_name=f"dataset {secret_value}",
            dataset_version="1.0.0",
            target_label="local",
            target_version=None,
            status="completed",
            started_at=STARTED_AT,
            completed_at=None,
        ),
        metrics=calculate_run_metrics(()),
        cases=(
            EvidenceCase(
                case_key="case_mask",
                status="failed",
                final_status=None,
                reviewer_note=f"note {secret_value}",
                assertions=(
                    EvidenceAssertion(
                        assertion_id="answer_exact",
                        assertion_type="exact_match",
                        status="failed",
                        detail=f"leaked {secret_value}",
                        required=True,
                        severity="high",
                        on_fail="fail",
                    ),
                ),
            ),
        ),
    )

    markdown = render_evidence_markdown(report)

    assert secret_value not in markdown
    assert "[masked:aws_access_key]" in markdown


def run_metadata() -> EvidenceRunMetadata:
    return EvidenceRunMetadata(
        run_id=1,
        dataset_name="dataset",
        dataset_version="1.0.0",
        target_label="local",
        target_version="mock-v1",
        status="completed",
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT,
    )
