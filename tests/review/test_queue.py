import asyncio
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from offline_llm_eval.dataset.repository import Dataset
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.review.queue import ReviewQueue, get_review_queue
from offline_llm_eval.run.case_result import CaseResultRecord
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus

STARTED_AT = datetime(2026, 5, 27, 13, 0, 0)
COMPLETED_AT = datetime(2026, 5, 27, 13, 5, 0)
HEARTBEAT_AT = datetime(2026, 5, 27, 13, 4, 30)
REVIEWED_AT = datetime(2026, 5, 27, 13, 10, 0)


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def migrate_database(database_path: Path) -> None:
    command.upgrade(make_config(database_path), "head")


def create_test_engine(database_path: Path) -> AsyncEngine:
    return create_async_db_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")


def test_review_queueはneeds_reviewだけを未投入優先で返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    queue = asyncio.run(run_get_review_queue(database_path))

    assert queue is not None
    assert queue.run.run_id == 1
    assert [item.case_key for item in queue.items] == [
        "case_needs_review_unreviewed",
        "case_needs_review_reviewed",
    ]
    assert queue.items[0].status is CaseResultStatus.NEEDS_REVIEW
    assert queue.items[0].reviewer_verdict is None
    assert queue.items[0].final_status is None
    assert queue.items[1].reviewer_verdict == "approved"
    assert queue.items[1].reviewer_note == "human reviewed"
    assert queue.items[1].reviewed_at == REVIEWED_AT
    assert queue.items[1].final_status == "pass"


async def run_get_review_queue(database_path: Path) -> ReviewQueue | None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            run_id = await seed_review_queue_run(session)

        async with session_factory.begin() as session:
            return await get_review_queue(session, run_id)
    finally:
        await engine.dispose()


def test_review_queueはneeds_reviewがなければ空配列を返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    queue = asyncio.run(run_get_empty_review_queue(database_path))

    assert queue is not None
    assert queue.items == ()


async def run_get_empty_review_queue(database_path: Path) -> ReviewQueue | None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="empty_review_queue_dataset", dataset_version="1.0.0")
            session.add(dataset)
            await session.flush()
            run_id = await add_run(session, dataset_id=dataset.dataset_id)
            await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=run_id,
                case_key="case_pass",
                status=CaseResultStatus.PASS,
            )

        async with session_factory.begin() as session:
            return await get_review_queue(session, run_id)
    finally:
        await engine.dispose()


def test_review_queueはmissing_runならnoneを返す(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    queue = asyncio.run(run_get_missing_review_queue(database_path))

    assert queue is None


async def run_get_missing_review_queue(database_path: Path) -> ReviewQueue | None:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            return await get_review_queue(session, 999)
    finally:
        await engine.dispose()


async def seed_review_queue_run(session: AsyncSession) -> int:
    dataset = Dataset(name="review_queue_dataset", dataset_version="1.0.0")
    session.add(dataset)
    await session.flush()
    run_id = await add_run(session, dataset_id=dataset.dataset_id)
    await add_case_result(
        session,
        dataset_id=dataset.dataset_id,
        run_id=run_id,
        case_key="case_pass",
        status=CaseResultStatus.PASS,
    )
    await add_case_result(
        session,
        dataset_id=dataset.dataset_id,
        run_id=run_id,
        case_key="case_needs_review_reviewed",
        status=CaseResultStatus.NEEDS_REVIEW,
        reviewer_verdict="approved",
        reviewer_note="human reviewed",
        reviewed_at=REVIEWED_AT,
        final_status="pass",
    )
    await add_case_result(
        session,
        dataset_id=dataset.dataset_id,
        run_id=run_id,
        case_key="case_needs_review_unreviewed",
        status=CaseResultStatus.NEEDS_REVIEW,
    )
    return run_id


async def add_run(session: AsyncSession, *, dataset_id: int) -> int:
    run = RunRecord(
        dataset_id=dataset_id,
        target_label="local",
        target_version="mock-v1",
        status=RunStatus.COMPLETED.value,
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT,
        last_heartbeat_at=HEARTBEAT_AT,
    )
    session.add(run)
    await session.flush()
    return run.run_id


async def add_case_result(
    session: AsyncSession,
    *,
    dataset_id: int,
    run_id: int,
    case_key: str,
    status: CaseResultStatus,
    reviewer_verdict: str | None = None,
    reviewer_note: str | None = None,
    reviewed_at: datetime | None = None,
    final_status: str | None = None,
) -> None:
    case_id = await insert_case(session, dataset_id=dataset_id, case_key=case_key)
    session.add(
        CaseResultRecord(
            run_id=run_id,
            case_id=case_id,
            case_key=case_key,
            status=status.value,
            reviewer_verdict=reviewer_verdict,
            reviewer_note=reviewer_note,
            reviewed_at=reviewed_at,
            final_status=final_status,
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
