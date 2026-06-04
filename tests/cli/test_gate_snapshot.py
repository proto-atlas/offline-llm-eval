import asyncio
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from offline_llm_eval.cli.gate_config_schema import GateConfigSchema
from offline_llm_eval.cli.gate_rules import (
    GateRulesEvaluation,
    GateRulesVerdict,
    ThresholdCriterion,
    ThresholdEvaluation,
    ThresholdStatus,
)
from offline_llm_eval.cli.gate_snapshot import (
    GateSnapshotWarning,
    save_gate_snapshot,
)
from offline_llm_eval.dataset.repository import Dataset
from offline_llm_eval.db import create_async_db_engine, create_session_factory
from offline_llm_eval.run.heartbeat import RunStatus
from offline_llm_eval.run.repository import RunRepository, RunSnapshot

STARTED_AT = datetime(2026, 5, 27, 12, 0, 0)
COMPLETED_AT = datetime(2026, 5, 27, 12, 5, 0)
EVALUATED_AT = datetime(2026, 5, 27, 12, 10, 0)
SECOND_EVALUATED_AT = datetime(2026, 5, 27, 12, 20, 0)


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def migrate_database(database_path: Path) -> None:
    command.upgrade(make_config(database_path), "head")


def create_test_engine(database_path: Path) -> AsyncEngine:
    return create_async_db_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")


async def create_dataset(session: AsyncSession) -> int:
    dataset = Dataset(name="gate_snapshot_dataset", dataset_version="1.0.0")
    session.add(dataset)
    await session.flush()
    return dataset.dataset_id


def test_gate_snapshotはrunにconfigとresultを保存する(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    snapshot, warnings = asyncio.run(run_save_gate_snapshot(database_path))

    assert warnings == ()
    assert snapshot.gate_config_snapshot_json == {
        "pass_rate_min": 0.9,
        "high_severity_must_pass": False,
        "fail_on_secret_leak": False,
    }
    assert snapshot.gate_result_json == {
        "verdict": "failed",
        "evaluated_at": "2026-05-27T12:10:00",
        "criteria": [
            {
                "name": "pass_rate_min",
                "status": "failed",
                "actual": 0.8,
                "threshold": 0.9,
                "reason": None,
            }
        ],
    }


async def run_save_gate_snapshot(
    database_path: Path,
) -> tuple[RunSnapshot, tuple[GateSnapshotWarning, ...]]:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            run_id = await create_completed_run(session)

        async with session_factory.begin() as session:
            result = await save_gate_snapshot(
                session,
                run_id=run_id,
                config=GateConfigSchema(pass_rate_min=0.9),
                evaluation=failed_threshold_evaluation(),
                evaluated_at=EVALUATED_AT,
            )
            assert result is not None
            return result.run, result.warnings
    finally:
        await engine.dispose()


def test_gate_snapshotは2回目保存でwarning付き上書きする(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)

    snapshot, warnings = asyncio.run(run_overwrite_gate_snapshot(database_path))

    assert warnings == (GateSnapshotWarning.GATE_SNAPSHOT_OVERWRITTEN,)
    assert snapshot.gate_config_snapshot_json == {
        "fail_rate_max": 0.1,
        "high_severity_must_pass": False,
        "fail_on_secret_leak": False,
    }
    assert snapshot.gate_result_json is not None
    assert snapshot.gate_result_json["verdict"] == "passed"
    assert snapshot.gate_result_json["evaluated_at"] == "2026-05-27T12:20:00"


async def run_overwrite_gate_snapshot(
    database_path: Path,
) -> tuple[RunSnapshot, tuple[GateSnapshotWarning, ...]]:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            run_id = await create_completed_run(session)
            first = await save_gate_snapshot(
                session,
                run_id=run_id,
                config=GateConfigSchema(pass_rate_min=0.9),
                evaluation=failed_threshold_evaluation(),
                evaluated_at=EVALUATED_AT,
            )
            assert first is not None

        async with session_factory.begin() as session:
            second = await save_gate_snapshot(
                session,
                run_id=run_id,
                config=GateConfigSchema(fail_rate_max=0.1),
                evaluation=passed_evaluation(),
                evaluated_at=SECOND_EVALUATED_AT,
            )
            assert second is not None
            return second.run, second.warnings
    finally:
        await engine.dispose()


async def create_completed_run(session: AsyncSession) -> int:
    dataset_id = await create_dataset(session)
    repository = RunRepository(session)
    created = await repository.create_run(
        dataset_id=dataset_id,
        target_label="local",
        started_at=STARTED_AT,
    )
    completed = await repository.complete_run(
        created.run_id,
        completed_at=COMPLETED_AT,
    )
    assert completed is not None
    assert completed.status is RunStatus.COMPLETED
    return completed.run_id


def failed_threshold_evaluation() -> GateRulesEvaluation:
    return GateRulesEvaluation(
        verdict=GateRulesVerdict.FAILED,
        high_severity_must_pass=None,
        fail_on_secret_leak=None,
        thresholds=(
            ThresholdEvaluation(
                criterion=ThresholdCriterion.PASS_RATE_MIN,
                status=ThresholdStatus.FAILED,
                actual=0.8,
                threshold=0.9,
            ),
        ),
    )


def passed_evaluation() -> GateRulesEvaluation:
    return GateRulesEvaluation(
        verdict=GateRulesVerdict.PASSED,
        high_severity_must_pass=None,
        fail_on_secret_leak=None,
        thresholds=(),
    )
