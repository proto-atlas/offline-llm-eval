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
from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.main import create_app
from offline_llm_eval.run.case_result import CaseResultRecord
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus
from offline_llm_eval.runner.error_type import ErrorType

BASELINE_STARTED_AT = datetime(2026, 5, 27, 10, 0, 0)
CURRENT_STARTED_AT = datetime(2026, 5, 27, 11, 0, 0)
COMPLETED_AT = datetime(2026, 5, 27, 11, 5, 0)
HEARTBEAT_AT = datetime(2026, 5, 27, 11, 4, 30)
BEFORE_BASELINE = datetime(2026, 5, 27, 9, 0, 0)
AFTER_BASELINE = datetime(2026, 5, 27, 10, 30, 0)


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


def test_diff_apiはbaseline比較結果を返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    current_run_id = asyncio.run(seed_diff_runs(database_path))
    client = create_test_client(database_path)

    response = client.get(f"/api/runs/{current_run_id}/diff", params={"baseline": "run:1"})

    assert response.status_code == 200
    body = response.json()
    assert body["baseline_run"]["run_id"] == 1
    assert body["current_run"]["run_id"] == current_run_id
    assert body["case_keys"] == {
        "shared": ["case_changed", "case_same"],
        "added": ["case_added"],
        "removed": ["case_removed"],
    }
    assert body["metrics"]["baseline"]["total"] == 3
    assert body["metrics"]["current"]["skipped"] == 1
    assert body["classifications"] == [
        {"classification": "changed_case", "case_key": "case_changed", "assertion_id": None},
        {"classification": "added_case", "case_key": "case_added", "assertion_id": None},
        {"classification": "removed_case", "case_key": "case_removed", "assertion_id": None},
        {
            "classification": "removed_assertion",
            "case_key": "case_changed",
            "assertion_id": "assertion_removed",
        },
    ]
    assert body["warnings"] == [
        {
            "code": "case_updated_after_baseline",
            "case_key": "case_changed",
            "baseline_run_id": 1,
            "baseline_started_at": "2026-05-27T10:00:00",
            "case_updated_at": "2026-05-27T10:30:00",
        }
    ]
    assert body["error_types"] == {
        "baseline": {
            "provider_error": 1,
            "overloaded": 0,
            "unknown_error": 0,
            "response_mode_mismatch": 0,
        },
        "current": {
            "provider_error": 0,
            "overloaded": 1,
            "unknown_error": 0,
            "response_mode_mismatch": 0,
        },
        "delta": {
            "provider_error": -1,
            "overloaded": 1,
            "unknown_error": 0,
            "response_mode_mismatch": 0,
        },
    }


def test_diff_apiはbaseline不在ならbaseline_not_foundを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    current_run_id = asyncio.run(seed_current_run_only(database_path))
    client = create_test_client(database_path)

    response = client.get(f"/api/runs/{current_run_id}/diff", params={"baseline": "run:999"})

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "baseline_not_found",
            "message": "基準実行が見つかりません。",
            "extra": {"baseline_spec": "run:999"},
        }
    }


def test_diff_apiはrunning_baselineならbaseline_in_progressを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    current_run_id = asyncio.run(seed_running_baseline(database_path))
    client = create_test_client(database_path)

    response = client.get(f"/api/runs/{current_run_id}/diff", params={"baseline": "run:1"})

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "code": "baseline_in_progress",
            "message": "基準実行が実行中のため差分を作成できません。",
            "extra": {"run_id": 1},
        }
    }


def test_diff_apiは不正baseline指定のsecret形状文字列を伏せる(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    current_run_id = asyncio.run(seed_current_run_only(database_path))
    client = create_test_client(database_path)
    secret_like = "".join(("AKIA", "12345678", "90ABCDEF"))

    response = client.get(f"/api/runs/{current_run_id}/diff", params={"baseline": secret_like})

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["extra"]["errors"][0]["msg"] == "基準実行の指定が不正です: [masked:aws_access_key]"
    assert secret_like not in error["extra"]["errors"][0]["msg"]


async def seed_diff_runs(database_path: Path) -> int:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="diff_api_dataset", dataset_version="1.0.0")
            session.add(dataset)
            await session.flush()

            case_ids = await seed_cases(session, dataset_id=dataset.dataset_id)
            baseline_run_id = await add_run(
                session,
                dataset_id=dataset.dataset_id,
                status=RunStatus.COMPLETED,
                started_at=BASELINE_STARTED_AT,
            )
            current_run_id = await add_run(
                session,
                dataset_id=dataset.dataset_id,
                status=RunStatus.COMPLETED,
                started_at=CURRENT_STARTED_AT,
            )
            assert baseline_run_id == 1

            add_case_result(
                session,
                run_id=baseline_run_id,
                case_id=case_ids["case_same"],
                case_key="case_same",
                status=CaseResultStatus.PASS,
            )
            add_case_result(
                session,
                run_id=baseline_run_id,
                case_id=case_ids["case_changed"],
                case_key="case_changed",
                status=CaseResultStatus.PASS,
            )
            add_case_result(
                session,
                run_id=baseline_run_id,
                case_id=case_ids["case_removed"],
                case_key="case_removed",
                status=CaseResultStatus.FAILED,
                error_type=ErrorType.PROVIDER_ERROR,
            )
            add_case_result(
                session,
                run_id=current_run_id,
                case_id=case_ids["case_same"],
                case_key="case_same",
                status=CaseResultStatus.PASS,
            )
            add_case_result(
                session,
                run_id=current_run_id,
                case_id=case_ids["case_changed"],
                case_key="case_changed",
                status=CaseResultStatus.FAILED,
            )
            add_case_result(
                session,
                run_id=current_run_id,
                case_id=case_ids["case_added"],
                case_key="case_added",
                status=CaseResultStatus.SKIPPED,
                error_type=ErrorType.OVERLOADED,
            )

            return current_run_id
    finally:
        await engine.dispose()


async def seed_current_run_only(database_path: Path) -> int:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="diff_api_missing_baseline_dataset", dataset_version="1.0.0")
            session.add(dataset)
            await session.flush()
            return await add_run(
                session,
                dataset_id=dataset.dataset_id,
                status=RunStatus.COMPLETED,
                started_at=CURRENT_STARTED_AT,
            )
    finally:
        await engine.dispose()


async def seed_running_baseline(database_path: Path) -> int:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="diff_api_running_baseline_dataset", dataset_version="1.0.0")
            session.add(dataset)
            await session.flush()
            baseline_run_id = await add_run(
                session,
                dataset_id=dataset.dataset_id,
                status=RunStatus.RUNNING,
                started_at=BASELINE_STARTED_AT,
                target_label="baseline",
            )
            assert baseline_run_id == 1
            return await add_run(
                session,
                dataset_id=dataset.dataset_id,
                status=RunStatus.COMPLETED,
                started_at=CURRENT_STARTED_AT,
                target_label="current",
            )
    finally:
        await engine.dispose()


async def seed_cases(session: AsyncSession, *, dataset_id: int) -> dict[str, int]:
    case_ids: dict[str, int] = {}
    for case_key, is_active, updated_at in (
        ("case_same", True, BEFORE_BASELINE),
        ("case_changed", True, AFTER_BASELINE),
        ("case_added", True, BEFORE_BASELINE),
        ("case_removed", False, BEFORE_BASELINE),
    ):
        result = await session.execute(
            text(
                """
                insert into evaluation_cases
                    (dataset_id, case_key, question, severity, tags_json, metadata_json,
                     is_active, updated_at)
                values
                    (:dataset_id, :case_key, 'Question?', 'medium', '[]', null,
                     :is_active, :updated_at)
                returning case_id
                """
            ),
            {
                "dataset_id": dataset_id,
                "case_key": case_key,
                "is_active": is_active,
                "updated_at": updated_at,
            },
        )
        case_ids[case_key] = result.scalar_one()

    await session.execute(
        text(
            """
            insert into assertions
                (case_id, id, assertion_type, expected_json, required, on_fail, severity,
                 is_active)
            values
                (:case_id, 'assertion_removed', 'exact_match', null, true, 'fail',
                 'medium', false)
            """
        ),
        {"case_id": case_ids["case_changed"]},
    )
    return case_ids


async def add_run(
    session: AsyncSession,
    *,
    dataset_id: int,
    status: RunStatus,
    started_at: datetime,
    target_label: str = "local",
) -> int:
    run = RunRecord(
        dataset_id=dataset_id,
        target_label=target_label,
        target_version="mock-v1",
        status=status.value,
        started_at=started_at,
        completed_at=COMPLETED_AT if status is not RunStatus.RUNNING else None,
        last_heartbeat_at=HEARTBEAT_AT,
        created_at=started_at,
        updated_at=started_at,
    )
    session.add(run)
    await session.flush()
    return run.run_id


def add_case_result(
    session: AsyncSession,
    *,
    run_id: int,
    case_id: int,
    case_key: str,
    status: CaseResultStatus,
    error_type: ErrorType | None = None,
) -> None:
    session.add(
        CaseResultRecord(
            run_id=run_id,
            case_id=case_id,
            case_key=case_key,
            status=status.value,
            error_type=error_type.value if error_type is not None else None,
        )
    )
