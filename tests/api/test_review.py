import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from offline_llm_eval.api.review import (
    ReviewFinalStatus,
    ReviewRecomputeResult,
    apply_review_transaction,
)
from offline_llm_eval.api.runs import get_database_session
from offline_llm_eval.dataset.repository import Dataset, JsonValue
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.evaluator.secret_pattern import SECRET_SCAN_ASSERTION_ID
from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.main import create_app
from offline_llm_eval.run.case_result import CaseResultRecord
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus

STARTED_AT = datetime(2026, 5, 27, 14, 0, 0)
COMPLETED_AT = datetime(2026, 5, 27, 14, 5, 0)
HEARTBEAT_AT = datetime(2026, 5, 27, 14, 4, 30)
REVIEWED_AT = datetime(2026, 5, 27, 14, 10, 0)


def join_parts(*parts: str) -> str:
    return "".join(parts)


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


def test_recompute(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))

    pass_result = asyncio.run(
        run_apply_review(
            database_path,
            run_id=run_id,
            case_key="case_needs_review_to_pass",
            final_status=ReviewFinalStatus.PASS,
        )
    )
    failed_result = asyncio.run(
        run_apply_review(
            database_path,
            run_id=run_id,
            case_key="case_needs_review_to_failed",
            final_status=ReviewFinalStatus.FAILED,
        )
    )
    client = create_test_client(database_path)

    summary_response = client.get(f"/api/runs/{run_id}")
    secret_scan_payload = asyncio.run(
        load_evaluator_results(database_path, case_key="case_needs_review_to_pass")
    )

    assert pass_result.case.effective_status is CaseResultStatus.PASS
    assert failed_result.case.effective_status is CaseResultStatus.FAILED
    assert failed_result.metrics.passed_count == 2
    assert failed_result.metrics.failed_count == 1
    assert failed_result.metrics.needs_review_count == 1
    assert summary_response.status_code == 200
    assert summary_response.json()["counts"] == {
        "total": 4,
        "executed": 4,
        "passed": 2,
        "failed": 1,
        "needs_review": 1,
        "skipped": 0,
    }
    assert summary_response.json()["rates"] == {
        "pass_rate": 0.5,
        "fail_rate": 0.25,
        "skipped_ratio": 0.0,
    }
    assert secret_scan_payload == [
        {
            "id": SECRET_SCAN_ASSERTION_ID,
            "status": "failed",
            "severity": "high",
            "required": True,
            "on_fail": "fail",
            "detail_code": "aws_access_key",
        }
    ]


def test_review_queue_endpointはneeds_reviewを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))
    client = create_test_client(database_path)

    response = client.get(f"/api/runs/{run_id}/review-queue")

    assert response.status_code == 200
    assert response.json() == {
        "run": {
            "run_id": run_id,
            "dataset_id": 1,
            "target_label": "local",
            "target_version": "mock-v1",
            "status": "completed",
        },
        "items": [
            {
                "case_key": "case_needs_review_to_failed",
                "status": "needs_review",
                "reviewer_verdict": None,
                "reviewer_note": None,
                "reviewed_at": None,
                "final_status": None,
            },
            {
                "case_key": "case_needs_review_to_pass",
                "status": "needs_review",
                "reviewer_verdict": None,
                "reviewer_note": None,
                "reviewed_at": None,
                "final_status": None,
            },
            {
                "case_key": "case_needs_review_unreviewed",
                "status": "needs_review",
                "reviewer_verdict": None,
                "reviewer_note": None,
                "reviewed_at": None,
                "final_status": None,
            },
        ],
    }


def test_review_queue_endpointは保存済みreviewer値を返却時にもmaskする(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))
    client = create_test_client(database_path)
    secret_value = join_parts("sk-proj-", "abcdefghijklmnopqrstuvwxyz")
    asyncio.run(
        save_raw_review_fields(
            database_path,
            case_key="case_needs_review_to_pass",
            reviewer_verdict=f"verdict {secret_value}",
            reviewer_note=f"note {secret_value}",
        )
    )

    response = client.get(f"/api/runs/{run_id}/review-queue")

    assert response.status_code == 200
    queue_item = next(
        item for item in response.json()["items"] if item["case_key"] == "case_needs_review_to_pass"
    )
    assert queue_item["reviewer_verdict"] == "verdict [masked:openai_api_key]"
    assert queue_item["reviewer_note"] == "note [masked:openai_api_key]"
    assert secret_value not in str(response.json())


def test_review_queue_endpointは保存済みverdictが伏せ字化後に長すぎたら汎用maskで返す(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))
    client = create_test_client(database_path)
    secret_values = (
        join_parts("AKIA", "12345678", "90ABCDEF"),
        join_parts("AKIA", "22345678", "90ABCDEF"),
        join_parts("AKIA", "32345678", "90ABCDEF"),
    )
    asyncio.run(
        save_raw_review_fields(
            database_path,
            case_key="case_needs_review_to_pass",
            reviewer_verdict=" ".join(secret_values),
            reviewer_note="human reviewed",
        )
    )

    response = client.get(f"/api/runs/{run_id}/review-queue")

    assert response.status_code == 200
    queue_item = next(
        item for item in response.json()["items"] if item["case_key"] == "case_needs_review_to_pass"
    )
    assert queue_item["reviewer_verdict"] == "[masked:reviewer_verdict]"
    assert all(secret_value not in str(response.json()) for secret_value in secret_values)


def test_review_queue_endpointはmissing_runならrun_not_foundを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    client = create_test_client(database_path)

    response = client.get("/api/runs/999/review-queue")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "run_not_found",
            "message": "実行が見つかりません。",
            "extra": {"run_id": 999},
        },
    }


def test_patch_review_endpointはreviewを保存して再集計する(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))
    client = create_test_client(database_path)

    response = client.patch(
        f"/api/runs/{run_id}/cases/case_needs_review_to_pass/review",
        json={
            "reviewer_verdict": "approved",
            "final_status": "pass",
            "reviewer_note": "human reviewed",
        },
    )
    summary_response = client.get(f"/api/runs/{run_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["case"] == {
        "case_result_id": 2,
        "case_key": "case_needs_review_to_pass",
        "status": "needs_review",
        "effective_status": "pass",
    }
    assert body["review"]["reviewer_verdict"] == "approved"
    assert body["review"]["reviewer_note"] == "human reviewed"
    assert body["review"]["final_status"] == "pass"
    assert isinstance(body["review"]["reviewed_at"], str)
    assert body["counts"] == {
        "total": 4,
        "executed": 4,
        "passed": 2,
        "failed": 0,
        "needs_review": 2,
        "skipped": 0,
    }
    assert body["rates"] == {
        "pass_rate": 0.5,
        "fail_rate": 0.0,
        "skipped_ratio": 0.0,
    }
    assert summary_response.status_code == 200
    assert summary_response.json()["counts"] == body["counts"]
    assert summary_response.json()["rates"] == body["rates"]


def test_patch_review_endpointはreviewer_noteのsecretをmaskして保存する(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))
    client = create_test_client(database_path)
    secret_value = join_parts("sk-proj-", "abcdefghijklmnopqrstuvwxyz")

    response = client.patch(
        f"/api/runs/{run_id}/cases/case_needs_review_to_pass/review",
        json={
            "reviewer_verdict": "approved",
            "final_status": "pass",
            "reviewer_note": f"note includes {secret_value}",
        },
    )
    queue_response = client.get(f"/api/runs/{run_id}/review-queue")
    case_response = client.get(f"/api/runs/{run_id}/cases/case_needs_review_to_pass")
    stored_note = asyncio.run(
        load_reviewer_note(database_path, case_key="case_needs_review_to_pass")
    )

    assert response.status_code == 200
    assert response.json()["review"]["reviewer_note"] == "note includes [masked:openai_api_key]"
    assert queue_response.status_code == 200
    queue_item = next(
        item
        for item in queue_response.json()["items"]
        if item["case_key"] == "case_needs_review_to_pass"
    )
    assert queue_item["reviewer_note"] == "note includes [masked:openai_api_key]"
    assert case_response.status_code == 200
    assert case_response.json()["review"]["reviewer_note"] == (
        "note includes [masked:openai_api_key]"
    )
    assert stored_note == "note includes [masked:openai_api_key]"
    assert secret_value not in str(response.json())
    assert secret_value not in str(queue_response.json())
    assert secret_value not in str(case_response.json())


def test_patch_review_endpointはreviewer_verdictのsecretをmaskして保存する(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))
    client = create_test_client(database_path)
    secret_value = join_parts("sk-proj-", "abcdefghijklmnopqrstuvwxyz")

    response = client.patch(
        f"/api/runs/{run_id}/cases/case_needs_review_to_pass/review",
        json={
            "reviewer_verdict": f"verdict {secret_value}",
            "final_status": "pass",
            "reviewer_note": "human reviewed",
        },
    )
    queue_response = client.get(f"/api/runs/{run_id}/review-queue")
    case_response = client.get(f"/api/runs/{run_id}/cases/case_needs_review_to_pass")
    stored_verdict = asyncio.run(
        load_reviewer_verdict(database_path, case_key="case_needs_review_to_pass")
    )

    assert response.status_code == 200
    assert response.json()["review"]["reviewer_verdict"] == ("verdict [masked:openai_api_key]")
    assert queue_response.status_code == 200
    queue_item = next(
        item
        for item in queue_response.json()["items"]
        if item["case_key"] == "case_needs_review_to_pass"
    )
    assert queue_item["reviewer_verdict"] == "verdict [masked:openai_api_key]"
    assert case_response.status_code == 200
    assert case_response.json()["review"]["reviewer_verdict"] == ("verdict [masked:openai_api_key]")
    assert stored_verdict == "verdict [masked:openai_api_key]"
    assert secret_value not in str(response.json())
    assert secret_value not in str(queue_response.json())
    assert secret_value not in str(case_response.json())


def test_patch_review_endpointはreviewer_verdictが64文字を超えたら422を返す(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))
    client = create_test_client(database_path)
    secret_value = join_parts("AKIA", "12345678", "90ABCDEF")

    response = client.patch(
        f"/api/runs/{run_id}/cases/case_needs_review_to_pass/review",
        json={
            "reviewer_verdict": f"{secret_value}{'x' * 45}",
            "final_status": "pass",
            "reviewer_note": "human reviewed",
        },
    )

    assert response.status_code == 422
    assert "input" not in str(response.json())
    assert secret_value not in str(response.json())


def test_patch_review_endpointは空のreviewer_verdictなら422を返す(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))
    client = create_test_client(database_path)

    response = client.patch(
        f"/api/runs/{run_id}/cases/case_needs_review_to_pass/review",
        json={
            "reviewer_verdict": "",
            "final_status": "pass",
            "reviewer_note": "human reviewed",
        },
    )

    assert response.status_code == 422


def test_patch_review_endpointは64文字のreviewer_verdictを保存する(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))
    client = create_test_client(database_path)
    reviewer_verdict = "x" * 64

    response = client.patch(
        f"/api/runs/{run_id}/cases/case_needs_review_to_pass/review",
        json={
            "reviewer_verdict": reviewer_verdict,
            "final_status": "pass",
            "reviewer_note": "human reviewed",
        },
    )

    assert response.status_code == 200
    assert response.json()["review"]["reviewer_verdict"] == reviewer_verdict


def test_patch_review_endpointは伏せ字化後に長すぎるverdictを汎用maskで保存する(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))
    client = create_test_client(database_path)
    secret_values = (
        join_parts("AKIA", "12345678", "90ABCDEF"),
        join_parts("AKIA", "22345678", "90ABCDEF"),
        join_parts("AKIA", "32345678", "90ABCDEF"),
    )
    reviewer_verdict = " ".join(secret_values)

    response = client.patch(
        f"/api/runs/{run_id}/cases/case_needs_review_to_pass/review",
        json={
            "reviewer_verdict": reviewer_verdict,
            "final_status": "pass",
            "reviewer_note": "human reviewed",
        },
    )
    stored_verdict = asyncio.run(
        load_reviewer_verdict(database_path, case_key="case_needs_review_to_pass")
    )

    assert response.status_code == 200
    assert response.json()["review"]["reviewer_verdict"] == "[masked:reviewer_verdict]"
    assert stored_verdict == "[masked:reviewer_verdict]"
    assert all(secret_value not in str(response.json()) for secret_value in secret_values)


def test_patch_review_endpointは伏せ字化後65文字のverdictを汎用maskで保存する(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))
    client = create_test_client(database_path)
    secret_values = (
        join_parts("AKIA", "12345678", "90ABCDEF"),
        join_parts("AKIA", "22345678", "90ABCDEF"),
    )
    reviewer_verdict = f"{secret_values[0]} {secret_values[1]} {'x' * 18}"

    response = client.patch(
        f"/api/runs/{run_id}/cases/case_needs_review_to_pass/review",
        json={
            "reviewer_verdict": reviewer_verdict,
            "final_status": "pass",
            "reviewer_note": "human reviewed",
        },
    )
    stored_verdict = asyncio.run(
        load_reviewer_verdict(database_path, case_key="case_needs_review_to_pass")
    )

    assert response.status_code == 200
    assert response.json()["review"]["reviewer_verdict"] == "[masked:reviewer_verdict]"
    assert stored_verdict == "[masked:reviewer_verdict]"
    assert all(secret_value not in str(response.json()) for secret_value in secret_values)


def test_patch_review_endpointはreviewer_noteが4000文字を超えたら422を返す(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))
    client = create_test_client(database_path)
    secret_value = join_parts("AKIA", "82345678", "90ABCDEF")

    response = client.patch(
        f"/api/runs/{run_id}/cases/case_needs_review_to_pass/review",
        json={
            "reviewer_verdict": "approved",
            "final_status": "pass",
            "reviewer_note": f"{'x' * 4001}{secret_value}",
        },
    )

    assert response.status_code == 422
    assert "input" not in str(response.json())
    assert secret_value not in str(response.json())


def test_patch_review_endpointはmissing_runならrun_not_foundを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    client = create_test_client(database_path)

    response = client.patch(
        "/api/runs/999/cases/case_1/review",
        json={"reviewer_verdict": "approved", "final_status": "pass"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "run_not_found",
            "message": "実行が見つかりません。",
            "extra": {"run_id": 999},
        },
    }


def test_patch_review_endpointはmissing_caseならcase_not_foundを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(seed_review_run(database_path))
    client = create_test_client(database_path)

    response = client.patch(
        f"/api/runs/{run_id}/cases/missing/review",
        json={"reviewer_verdict": "approved", "final_status": "pass"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "case_not_found",
            "message": "ケースが見つかりません。",
            "extra": {"run_id": run_id, "case_key": "missing"},
        },
    }


async def run_apply_review(
    database_path: Path,
    *,
    run_id: int,
    case_key: str,
    final_status: ReviewFinalStatus,
) -> ReviewRecomputeResult:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            return await apply_review_transaction(
                session,
                run_id=run_id,
                case_key=case_key,
                reviewer_verdict=f"human_{final_status.value}",
                final_status=final_status,
                reviewer_note="human reviewed",
                reviewed_at=REVIEWED_AT,
            )
    finally:
        await engine.dispose()


async def seed_review_run(database_path: Path) -> int:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="review_api_dataset", dataset_version="1.0.0")
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

            await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=run.run_id,
                case_key="case_pass",
                status=CaseResultStatus.PASS,
            )
            await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=run.run_id,
                case_key="case_needs_review_to_pass",
                status=CaseResultStatus.NEEDS_REVIEW,
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
            await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=run.run_id,
                case_key="case_needs_review_to_failed",
                status=CaseResultStatus.NEEDS_REVIEW,
            )
            await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=run.run_id,
                case_key="case_needs_review_unreviewed",
                status=CaseResultStatus.NEEDS_REVIEW,
            )
            return run.run_id
    finally:
        await engine.dispose()


async def add_case_result(
    session: AsyncSession,
    *,
    dataset_id: int,
    run_id: int,
    case_key: str,
    status: CaseResultStatus,
    evaluator_results_json: JsonValue | None = None,
) -> None:
    case_id = await insert_case(session, dataset_id=dataset_id, case_key=case_key)
    session.add(
        CaseResultRecord(
            run_id=run_id,
            case_id=case_id,
            case_key=case_key,
            status=status.value,
            evaluator_results_json=evaluator_results_json,
        )
    )


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


async def load_evaluator_results(database_path: Path, *, case_key: str) -> JsonValue | None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            result = await session.execute(
                select(CaseResultRecord.evaluator_results_json).where(
                    CaseResultRecord.case_key == case_key
                )
            )
            return result.scalar_one()
    finally:
        await engine.dispose()


async def load_reviewer_note(database_path: Path, *, case_key: str) -> str | None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            result = await session.execute(
                select(CaseResultRecord.reviewer_note).where(CaseResultRecord.case_key == case_key)
            )
            return result.scalar_one()
    finally:
        await engine.dispose()


async def load_reviewer_verdict(database_path: Path, *, case_key: str) -> str | None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            result = await session.execute(
                select(CaseResultRecord.reviewer_verdict).where(
                    CaseResultRecord.case_key == case_key
                )
            )
            return result.scalar_one()
    finally:
        await engine.dispose()


async def save_raw_review_fields(
    database_path: Path,
    *,
    case_key: str,
    reviewer_verdict: str,
    reviewer_note: str,
) -> None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            await session.execute(
                text(
                    """
                    update case_results
                    set reviewer_verdict = :reviewer_verdict,
                        reviewer_note = :reviewer_note
                    where case_key = :case_key
                    """
                ),
                {
                    "case_key": case_key,
                    "reviewer_verdict": reviewer_verdict,
                    "reviewer_note": reviewer_note,
                },
            )
    finally:
        await engine.dispose()
