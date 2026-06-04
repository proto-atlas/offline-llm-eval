import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from offline_llm_eval.api.evidence import build_evidence_report
from offline_llm_eval.api.runs import get_database_session
from offline_llm_eval.dataset.repository import Dataset, JsonValue
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.evaluator.secret_pattern import SECRET_SCAN_ASSERTION_ID
from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.evidence.markdown import EvidenceReport
from offline_llm_eval.main import create_app
from offline_llm_eval.run.case_result import AssertionResultRecord, CaseResultRecord
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus

STARTED_AT = datetime(2026, 5, 27, 16, 0, 0)
COMPLETED_AT = datetime(2026, 5, 27, 16, 5, 0)
HEARTBEAT_AT = datetime(2026, 5, 27, 16, 4, 30)
AWS_ACCESS_KEY_VALUE = "".join(("AKIA", "12345678", "90ABCDEF"))


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def migrate_database(database_path: Path) -> None:
    command.upgrade(make_config(database_path), "head")


def create_test_engine(database_path: Path) -> AsyncEngine:
    return create_async_db_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")


def create_test_client(database_path: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_database_session] = make_session_override(database_path)
    return TestClient(app)


def make_session_override(database_path: Path) -> Callable[[], AsyncIterator[AsyncSession]]:
    async def override() -> AsyncIterator[AsyncSession]:
        engine = create_test_engine(database_path)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                yield session
        finally:
            await engine.dispose()

    return override


def test_evidence_apiはmarkdownを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_evidence_run(database_path))
    client = create_test_client(database_path)

    response = client.get(f"/api/runs/{run_id}/evidence")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    markdown = response.text
    assert "# 実行証跡" in markdown
    assert "- dataset: evidence_dataset@1.0.0" in markdown
    assert "- target_label: local" in markdown
    assert "- total: 3" in markdown
    assert "- passed: 2" in markdown
    assert "- failed: 1" in markdown
    assert "- needs_review: 0" in markdown
    assert "### case_failed" in markdown
    assert "answer_exact (exact_match, status=failed" in markdown
    assert "### case_warning" in markdown
    assert "optional_keyword (keyword_any, status=failed" in markdown
    assert "- case_failed: __secret_scan__ (aws_access_key)" in markdown
    assert "## 主張しない範囲（not_claimed）" in markdown


def test_evidence_apiはsecret値をmaskする(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_evidence_run(database_path))
    client = create_test_client(database_path)

    response = client.get(f"/api/runs/{run_id}/evidence")

    assert response.status_code == 200
    assert AWS_ACCESS_KEY_VALUE not in response.text
    assert "[masked:aws_access_key]" in response.text
    report = asyncio.run(load_evidence_report(database_path, run_id=run_id))
    failed_case = next(case for case in report.cases if case.case_key == "case_failed")
    assert failed_case.reviewer_note == "review note [masked:aws_access_key]"


def test_evidence_apiはmissing_runならrun_not_foundを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    client = create_test_client(database_path)

    response = client.get("/api/runs/999/evidence")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "run_not_found",
            "message": "実行が見つかりません。",
            "extra": {"run_id": 999},
        },
    }


async def seed_evidence_run(database_path: Path) -> int:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="evidence_dataset", dataset_version="1.0.0")
            session.add(dataset)
            await session.flush()
            run = RunRecord(
                dataset_id=dataset.dataset_id,
                target_label="local",
                target_version="mock-v1",
                status=RunStatus.COMPLETED.value,
                started_at=STARTED_AT,
                completed_at=COMPLETED_AT,
                last_heartbeat_at=HEARTBEAT_AT,
            )
            session.add(run)
            await session.flush()

            failed_case = await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=run.run_id,
                case_key="case_failed",
                status=CaseResultStatus.FAILED,
                reviewer_note=f"review note {AWS_ACCESS_KEY_VALUE}",
                evaluator_results_json=[
                    {
                        "id": SECRET_SCAN_ASSERTION_ID,
                        "status": "failed",
                        "severity": "high",
                        "required": True,
                        "on_fail": "fail",
                        "detail_code": "aws_access_key",
                    }
                ],
            )
            warning_case = await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=run.run_id,
                case_key="case_warning",
                status=CaseResultStatus.PASS,
            )
            await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=run.run_id,
                case_key="case_reviewed",
                status=CaseResultStatus.NEEDS_REVIEW,
                final_status="pass",
            )
            session.add_all(
                [
                    AssertionResultRecord(
                        case_result_id=failed_case.case_result_id,
                        assertion_db_id=101,
                        assertion_id="answer_exact",
                        assertion_type="exact_match",
                        status="failed",
                        detail=f"exact_match_mismatch {AWS_ACCESS_KEY_VALUE}",
                        matched_value_json="actual",
                        expected_json="expected",
                        required=True,
                        severity="high",
                        on_fail="fail",
                    ),
                    AssertionResultRecord(
                        case_result_id=warning_case.case_result_id,
                        assertion_db_id=102,
                        assertion_id="optional_keyword",
                        assertion_type="keyword_any",
                        status="failed",
                        detail="missing_optional_keyword",
                        matched_value_json=["actual"],
                        expected_json=["optional"],
                        required=False,
                        severity="medium",
                        on_fail="warn",
                    ),
                ]
            )
            return run.run_id
    finally:
        await engine.dispose()


async def load_evidence_report(database_path: Path, *, run_id: int) -> EvidenceReport:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            report = await build_evidence_report(session, run_id)
            assert report is not None
            return report
    finally:
        await engine.dispose()


async def add_case_result(
    session: AsyncSession,
    *,
    dataset_id: int,
    run_id: int,
    case_key: str,
    status: CaseResultStatus,
    reviewer_note: str | None = None,
    final_status: str | None = None,
    evaluator_results_json: JsonValue | None = None,
) -> CaseResultRecord:
    case_id = await insert_case(session, dataset_id=dataset_id, case_key=case_key)
    case_result = CaseResultRecord(
        run_id=run_id,
        case_id=case_id,
        case_key=case_key,
        status=status.value,
        evaluator_results_json=evaluator_results_json,
        reviewer_note=reviewer_note,
        final_status=final_status,
    )
    session.add(case_result)
    await session.flush()
    return case_result


async def insert_case(session: AsyncSession, *, dataset_id: int, case_key: str) -> int:
    result = await session.execute(
        text(
            """
            insert into evaluation_cases
                (dataset_id, case_key, question, severity, tags_json, metadata_json)
            values
                (:dataset_id, :case_key, 'Question?', 'medium', '[]', null)
            returning case_id
            """
        ),
        {"dataset_id": dataset_id, "case_key": case_key},
    )
    return int(result.scalar_one())
