import asyncio
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from offline_llm_eval.cli.gate import main, parse_args
from offline_llm_eval.dataset.assertion_model import AssertionOnFail, AssertionSeverity
from offline_llm_eval.dataset.repository import Dataset, JsonObject
from offline_llm_eval.db import DATABASE_URL_ENV, create_async_db_engine, create_session_factory
from offline_llm_eval.evaluator.assertions import AssertionEvaluationStatus
from offline_llm_eval.evaluator.results_schema import (
    build_secret_scan_pseudo_result,
    dump_pseudo_evaluator_result,
)
from offline_llm_eval.evaluator.secret_pattern import SecretScanStatus
from offline_llm_eval.evaluator.status import CaseResultStatus
from offline_llm_eval.run.case_result import AssertionResultRecord, CaseResultRecord
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus
from offline_llm_eval.run.repository import RunRepository, RunSnapshot

BASELINE_STARTED_AT = datetime(2026, 5, 27, 11, 0, 0)
CURRENT_STARTED_AT = datetime(2026, 5, 27, 12, 0, 0)
COMPLETED_AT = datetime(2026, 5, 27, 12, 5, 0)
HEARTBEAT_AT = datetime(2026, 5, 27, 12, 4, 30)


@dataclass(frozen=True, slots=True)
class AssertionSeed:
    assertion_id: str
    status: AssertionEvaluationStatus
    required: bool = True
    severity: AssertionSeverity = AssertionSeverity.HIGH
    on_fail: AssertionOnFail = AssertionOnFail.FAIL


@dataclass(frozen=True, slots=True)
class CaseSeed:
    case_key: str
    status: CaseResultStatus
    severity: AssertionSeverity = AssertionSeverity.MEDIUM
    assertions: tuple[AssertionSeed, ...] = ()
    pseudo_results: tuple[JsonObject, ...] = ()


def make_config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path.as_posix()}")
    return config


def migrate_database(database_path: Path) -> None:
    command.upgrade(make_config(database_path), "head")


def create_test_engine(database_path: Path) -> AsyncEngine:
    return create_async_db_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")


def run_gate_cli(
    database_path: Path,
    argv: Sequence[str],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, str]:
    monkeypatch.setenv(DATABASE_URL_ENV, f"sqlite+aiosqlite:///{database_path.as_posix()}")
    stderr = StringIO()
    exit_code = main(argv, stderr=stderr)
    return exit_code, stderr.getvalue()


def test_gate_cliはpassedならexit0でsnapshotを保存する(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(
        seed_single_run(
            database_path,
            cases=(CaseSeed("case_pass", CaseResultStatus.PASS),),
        )
    )

    exit_code, stderr = run_gate_cli(
        database_path,
        ["--run", str(run_id), "--pass-rate-min", "1.0"],
        monkeypatch,
    )
    snapshot = asyncio.run(load_run_snapshot(database_path, run_id))

    assert exit_code == 0
    assert stderr == ""
    assert snapshot.gate_result_json is not None
    assert snapshot.gate_result_json["verdict"] == "passed"
    assert snapshot.gate_result_json["criteria"] == [
        {
            "name": "pass_rate_min",
            "status": "passed",
            "actual": 1.0,
            "threshold": 1.0,
            "reason": None,
        }
    ]


def test_gate_cliはfailedならexit1でsnapshotを保存する(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(
        seed_single_run(
            database_path,
            cases=(CaseSeed("case_failed", CaseResultStatus.FAILED),),
        )
    )

    exit_code, stderr = run_gate_cli(
        database_path,
        ["--run", str(run_id), "--pass-rate-min", "0.5"],
        monkeypatch,
    )
    snapshot = asyncio.run(load_run_snapshot(database_path, run_id))

    assert exit_code == 1
    assert stderr == ""
    assert snapshot.gate_result_json is not None
    assert snapshot.gate_result_json["verdict"] == "failed"
    assert snapshot.gate_result_json["criteria"] == [
        {
            "name": "pass_rate_min",
            "status": "failed",
            "actual": 0.0,
            "threshold": 0.5,
            "reason": None,
        }
    ]


def test_gate_cliはschema_errorならexit2を返す(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "app.db"
    config_path = tmp_path / "gate.yaml"
    config_path.write_text("pass_rate_min: 1.1\n", encoding="utf-8")

    exit_code, stderr = run_gate_cli(
        database_path,
        ["--run", "1", "--config", str(config_path)],
        monkeypatch,
    )

    assert exit_code == 2
    assert "validation_error: pass_rate_min:" in stderr


def test_gate_cliはboolean条件をfalseで上書きできる() -> None:
    options = parse_args(
        [
            "--run",
            "1",
            "--no-high-severity-must-pass",
            "--no-fail-on-secret-leak",
        ]
    )

    assert options.high_severity_must_pass is False
    assert options.fail_on_secret_leak is False


def test_gate_cliはbaselineなしならdelta条件をskipする(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(
        seed_single_run(
            database_path,
            cases=(CaseSeed("case_failed", CaseResultStatus.FAILED),),
        )
    )

    exit_code, stderr = run_gate_cli(
        database_path,
        ["--run", str(run_id), "--max-fail-rate-delta", "0.0"],
        monkeypatch,
    )
    snapshot = asyncio.run(load_run_snapshot(database_path, run_id))

    assert exit_code == 0
    assert stderr == ""
    assert snapshot.gate_result_json is not None
    assert snapshot.gate_result_json["verdict"] == "passed"
    assert snapshot.gate_result_json["criteria"] == [
        {
            "name": "max_fail_rate_delta",
            "status": "skipped",
            "actual": None,
            "threshold": 0.0,
            "reason": "baseline_not_available",
        }
    ]


def test_gate_cliは明示baselineが存在しなければexit2を返す(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(
        seed_single_run(
            database_path,
            cases=(CaseSeed("case_pass", CaseResultStatus.PASS),),
        )
    )

    exit_code, stderr = run_gate_cli(
        database_path,
        ["--run", str(run_id), "--baseline", "run:999", "--max-fail-rate-delta", "0.0"],
        monkeypatch,
    )

    assert exit_code == 2
    assert stderr == "baseline_not_found: 基準実行が見つかりません: baseline=run:999\n"


def test_gate_cliはpython_mでhelpを表示する() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "offline_llm_eval.cli.gate", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert result.returncode == 0
    assert "usage: offline-llm-eval-check" in result.stdout
    assert result.stderr == ""


def test_gate_cliはbaseline比較でexit1を返す(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    current_run_id = asyncio.run(seed_baseline_and_current_runs(database_path))

    exit_code, stderr = run_gate_cli(
        database_path,
        ["--run", str(current_run_id), "--max-fail-rate-delta", "0.0"],
        monkeypatch,
    )
    snapshot = asyncio.run(load_run_snapshot(database_path, current_run_id))

    assert exit_code == 1
    assert stderr == ""
    assert snapshot.gate_result_json is not None
    assert snapshot.gate_result_json["criteria"] == [
        {
            "name": "max_fail_rate_delta",
            "status": "failed",
            "actual": 1.0,
            "threshold": 0.0,
            "reason": None,
        }
    ]


def test_gate_cliは2回目保存でstderr_warningを出す(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(
        seed_single_run(
            database_path,
            cases=(CaseSeed("case_pass", CaseResultStatus.PASS),),
        )
    )

    first_exit_code, first_stderr = run_gate_cli(
        database_path,
        ["--run", str(run_id), "--pass-rate-min", "1.0"],
        monkeypatch,
    )
    second_exit_code, second_stderr = run_gate_cli(
        database_path,
        ["--run", str(run_id), "--pass-rate-min", "1.0"],
        monkeypatch,
    )

    assert first_exit_code == 0
    assert first_stderr == ""
    assert second_exit_code == 0
    assert second_stderr == "gate_snapshot_overwritten\n"


def test_gate_cliはsecret_leakをcaseから読み込む(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "app.db"
    migrate_database(database_path)
    run_id = asyncio.run(
        seed_single_run(
            database_path,
            cases=(
                CaseSeed(
                    "case_secret",
                    CaseResultStatus.FAILED,
                    pseudo_results=(
                        dump_pseudo_evaluator_result(
                            build_secret_scan_pseudo_result(
                                status=SecretScanStatus.FAILED,
                                detail_code="aws_access_key",
                            )
                        ),
                    ),
                ),
            ),
        )
    )

    exit_code, stderr = run_gate_cli(
        database_path,
        ["--run", str(run_id), "--fail-on-secret-leak"],
        monkeypatch,
    )
    snapshot = asyncio.run(load_run_snapshot(database_path, run_id))

    assert exit_code == 1
    assert stderr == ""
    assert snapshot.gate_result_json is not None
    assert snapshot.gate_result_json["criteria"] == [
        {
            "name": "fail_on_secret_leak",
            "status": "failed",
            "failures": [
                {
                    "case_key": "case_secret",
                    "assertion_id": "__secret_scan__",
                    "reason": "secret_scan_failed",
                }
            ],
        }
    ]


async def seed_single_run(
    database_path: Path,
    *,
    cases: Sequence[CaseSeed],
) -> int:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="gate_cli_dataset", dataset_version="1.0.0")
            session.add(dataset)
            await session.flush()
            run_id = await add_run(
                session,
                dataset_id=dataset.dataset_id,
                started_at=CURRENT_STARTED_AT,
            )
            for case in cases:
                await add_case_result(
                    session,
                    dataset_id=dataset.dataset_id,
                    run_id=run_id,
                    case=case,
                )
            return run_id
    finally:
        await engine.dispose()


async def seed_baseline_and_current_runs(database_path: Path) -> int:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            dataset = Dataset(name="gate_cli_baseline_dataset", dataset_version="1.0.0")
            session.add(dataset)
            await session.flush()
            baseline_run_id = await add_run(
                session,
                dataset_id=dataset.dataset_id,
                started_at=BASELINE_STARTED_AT,
            )
            current_run_id = await add_run(
                session,
                dataset_id=dataset.dataset_id,
                started_at=CURRENT_STARTED_AT,
            )
            await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=baseline_run_id,
                case=CaseSeed("case_main", CaseResultStatus.PASS),
            )
            await add_case_result(
                session,
                dataset_id=dataset.dataset_id,
                run_id=current_run_id,
                case=CaseSeed("case_main", CaseResultStatus.FAILED),
            )
            return current_run_id
    finally:
        await engine.dispose()


async def add_run(
    session: AsyncSession,
    *,
    dataset_id: int,
    started_at: datetime,
) -> int:
    run = RunRecord(
        dataset_id=dataset_id,
        target_label="local",
        target_version="mock-v1",
        status=RunStatus.COMPLETED.value,
        started_at=started_at,
        completed_at=COMPLETED_AT,
        last_heartbeat_at=HEARTBEAT_AT,
        created_at=started_at,
        updated_at=started_at,
    )
    session.add(run)
    await session.flush()
    return run.run_id


async def add_case_result(
    session: AsyncSession,
    *,
    dataset_id: int,
    run_id: int,
    case: CaseSeed,
) -> None:
    case_id = await insert_case(session, dataset_id=dataset_id, case=case)
    case_result = CaseResultRecord(
        run_id=run_id,
        case_id=case_id,
        case_key=case.case_key,
        status=case.status.value,
        evaluator_results_json=list(case.pseudo_results) if case.pseudo_results else None,
    )
    session.add(case_result)
    await session.flush()
    for assertion_index, assertion in enumerate(case.assertions, start=1):
        session.add(
            AssertionResultRecord(
                case_result_id=case_result.case_result_id,
                assertion_db_id=assertion_index,
                assertion_id=assertion.assertion_id,
                assertion_type="exact_match",
                status=assertion.status.value,
                detail=None,
                matched_value_json=None,
                expected_json=None,
                required=assertion.required,
                severity=assertion.severity.value,
                on_fail=assertion.on_fail.value,
            )
        )


async def insert_case(
    session: AsyncSession,
    *,
    dataset_id: int,
    case: CaseSeed,
) -> int:
    existing = await session.execute(
        text(
            """
            select case_id
            from evaluation_cases
            where dataset_id = :dataset_id and case_key = :case_key
            """
        ),
        {"dataset_id": dataset_id, "case_key": case.case_key},
    )
    existing_case_id = existing.scalar_one_or_none()
    if existing_case_id is not None:
        return int(existing_case_id)

    result = await session.execute(
        text(
            """
            insert into evaluation_cases
                (dataset_id, case_key, question, severity, tags_json, metadata_json)
            values
                (:dataset_id, :case_key, 'Question?', :severity, '[]', null)
            returning case_id
            """
        ),
        {
            "dataset_id": dataset_id,
            "case_key": case.case_key,
            "severity": case.severity.value,
        },
    )
    return int(result.scalar_one())


async def load_run_snapshot(database_path: Path, run_id: int) -> RunSnapshot:
    engine = create_test_engine(database_path)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory.begin() as session:
            snapshot = await RunRepository(session).get_run(run_id)
            assert snapshot is not None
            return snapshot
    finally:
        await engine.dispose()
