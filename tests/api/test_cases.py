import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from offline_llm_eval.api.runs import get_database_session
from offline_llm_eval.dataset.repository import Dataset
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.evaluator.secret_pattern import SECRET_SCAN_ASSERTION_ID
from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.main import create_app
from offline_llm_eval.run.case_result import AssertionResultRecord, CaseResultRecord
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus
from offline_llm_eval.runner.error_type import ErrorType

STARTED_AT = datetime(2026, 5, 27, 11, 0, 0)
REVIEWED_AT = datetime(2026, 5, 27, 11, 10, 0)
AWS_ACCESS_KEY_VALUE = "".join(("AKIA", "42345678", "90ABCDEF"))
MATCHED_AWS_ACCESS_KEY_VALUE = "".join(("AKIA", "52345678", "90ABCDEF"))
EXPECTED_AWS_ACCESS_KEY_VALUE = "".join(("AKIA", "62345678", "90ABCDEF"))
OPENAI_API_KEY_VALUE = "".join(("sk-proj-", "abcdefghijklmnopqrstuvwxyz"))


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


def test_case_detailは通常とpseudoを9fieldで返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    asyncio.run(seed_case_detail(database_path))
    client = create_test_client(database_path)

    response = client.get("/api/runs/1/cases/case_1")
    body = response.json()

    assert response.status_code == 200
    assert body == {
        "run": {
            "run_id": 1,
            "target_label": "local",
            "target_version": "mock-v1",
            "status": "completed",
        },
        "case": {
            "case_result_id": 1,
            "case_id": 1,
            "case_key": "case_1",
            "status": "failed",
            "error_type": "provider_error",
        },
        "review": {
            "reviewer_verdict": "failed [masked:openai_api_key]",
            "reviewer_note": "Needs provider fix. [masked:openai_api_key]",
            "reviewed_at": "2026-05-27T11:10:00",
            "final_status": "failed",
        },
        "assertions": [
            {
                "assertion_id": "answer_exact",
                "assertion_type": "exact_match",
                "status": "failed",
                "detail": "exact_match_mismatch [masked:aws_access_key]",
                "matched_value": {"[masked:aws_access_key]": "actual"},
                "expected": {"nested": {"[masked:aws_access_key]": "expected"}},
                "required": True,
                "severity": "high",
                "on_fail": "fail",
            },
            {
                "assertion_id": "safe_keyword",
                "assertion_type": "keyword_all",
                "status": "pass",
                "detail": None,
                "matched_value": ["safe"],
                "expected": ["safe"],
                "required": False,
                "severity": "medium",
                "on_fail": "warn",
            },
            {
                "assertion_id": SECRET_SCAN_ASSERTION_ID,
                "assertion_type": "secret_scan",
                "status": "failed",
                "detail": "aws_access_key",
                "matched_value": None,
                "expected": None,
                "required": True,
                "severity": "high",
                "on_fail": "fail",
            },
        ],
        "failed_assertions": [
            {
                "assertion_id": "answer_exact",
                "assertion_type": "exact_match",
                "status": "failed",
                "detail": "exact_match_mismatch [masked:aws_access_key]",
                "matched_value": {"[masked:aws_access_key]": "actual"},
                "expected": {"nested": {"[masked:aws_access_key]": "expected"}},
                "required": True,
                "severity": "high",
                "on_fail": "fail",
            },
            {
                "assertion_id": SECRET_SCAN_ASSERTION_ID,
                "assertion_type": "secret_scan",
                "status": "failed",
                "detail": "aws_access_key",
                "matched_value": None,
                "expected": None,
                "required": True,
                "severity": "high",
                "on_fail": "fail",
            },
        ],
    }
    assert AWS_ACCESS_KEY_VALUE not in str(body)
    assert MATCHED_AWS_ACCESS_KEY_VALUE not in str(body)
    assert EXPECTED_AWS_ACCESS_KEY_VALUE not in str(body)
    assert OPENAI_API_KEY_VALUE not in str(body)


async def seed_case_detail(database_path: Path) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="case_detail_dataset", dataset_version="1.0.0")
            session.add(dataset)
            await session.flush()
            case_id = await insert_evaluation_case(session, dataset_id=dataset.dataset_id)
            run = RunRecord(
                dataset_id=dataset.dataset_id,
                target_label="local",
                target_version="mock-v1",
                status=RunStatus.COMPLETED.value,
                started_at=STARTED_AT,
            )
            session.add(run)
            await session.flush()
            case_result = CaseResultRecord(
                run_id=run.run_id,
                case_id=case_id,
                case_key="case_1",
                status=CaseResultStatus.FAILED.value,
                error_type=ErrorType.PROVIDER_ERROR.value,
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
                reviewer_verdict=f"failed {OPENAI_API_KEY_VALUE}",
                reviewer_note=f"Needs provider fix. {OPENAI_API_KEY_VALUE}",
                reviewed_at=REVIEWED_AT,
                final_status="failed",
            )
            session.add(case_result)
            await session.flush()
            session.add_all(
                [
                    AssertionResultRecord(
                        case_result_id=case_result.case_result_id,
                        assertion_db_id=101,
                        assertion_id="answer_exact",
                        assertion_type="exact_match",
                        status="failed",
                        detail=f"exact_match_mismatch {AWS_ACCESS_KEY_VALUE}",
                        matched_value_json={MATCHED_AWS_ACCESS_KEY_VALUE: "actual"},
                        expected_json={"nested": {EXPECTED_AWS_ACCESS_KEY_VALUE: "expected"}},
                        required=True,
                        severity="high",
                        on_fail="fail",
                    ),
                    AssertionResultRecord(
                        case_result_id=case_result.case_result_id,
                        assertion_db_id=102,
                        assertion_id="safe_keyword",
                        assertion_type="keyword_all",
                        status="pass",
                        detail=None,
                        matched_value_json=["safe"],
                        expected_json=["safe"],
                        required=False,
                        severity="medium",
                        on_fail="warn",
                    ),
                ]
            )
    finally:
        await engine.dispose()


async def insert_evaluation_case(session: AsyncSession, *, dataset_id: int) -> int:
    await session.execute(
        text(
            """
            insert into evaluation_cases
                (dataset_id, case_key, question, severity, tags_json, metadata_json)
            values
                (:dataset_id, 'case_1', 'Question?', 'high', '[]', null)
            """
        ),
        {"dataset_id": dataset_id},
    )
    result = await session.execute(
        text(
            """
            select case_id
            from evaluation_cases
            where dataset_id = :dataset_id and case_key = 'case_1'
            """
        ),
        {"dataset_id": dataset_id},
    )
    return int(result.scalar_one())


def test_case_detailはmissing_runならrun_not_foundを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    client = create_test_client(database_path)

    response = client.get("/api/runs/999/cases/case_1")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "run_not_found",
            "message": "実行が見つかりません。",
            "extra": {"run_id": 999},
        },
    }


def test_case_detailはmissing_caseならcase_not_foundを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    asyncio.run(seed_run_without_case(database_path))
    client = create_test_client(database_path)

    response = client.get("/api/runs/1/cases/missing")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "case_not_found",
            "message": "ケースが見つかりません。",
            "extra": {"run_id": 1, "case_key": "missing"},
        },
    }


async def seed_run_without_case(database_path: Path) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="case_missing_dataset", dataset_version="1.0.0")
            session.add(dataset)
            await session.flush()
            session.add(
                RunRecord(
                    dataset_id=dataset.dataset_id,
                    target_label="local",
                    target_version=None,
                    status=RunStatus.COMPLETED.value,
                    started_at=STARTED_AT,
                )
            )
    finally:
        await engine.dispose()
