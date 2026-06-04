from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import Boolean, Column, Integer, MetaData, String, Table, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from offline_llm_eval.dataset.importer import (
    DatasetCaseInput,
    import_dataset_document,
    parse_dataset_import_document,
)
from offline_llm_eval.dataset.repository import DatasetRepository
from offline_llm_eval.run.heartbeat import RunRecord, RunStatus
from offline_llm_eval.run.stale_sweep import (
    DEFAULT_STALE_TIMEOUT,
    sweep_stale_runs_for_dataset_import,
)

METADATA = MetaData()
EVALUATION_CASES = Table(
    "evaluation_cases",
    METADATA,
    Column("case_id", Integer),
    Column("dataset_id", Integer),
    Column("case_key", String),
    Column("is_active", Boolean),
)
ASSERTIONS = Table(
    "assertions",
    METADATA,
    Column("assertion_db_id", Integer),
    Column("case_id", Integer),
    Column("id", String),
    Column("is_active", Boolean),
)


@dataclass(frozen=True, slots=True)
class DeactivatedAssertion:
    case_key: str
    assertion_id: str


@dataclass(frozen=True, slots=True)
class FullReplacementImportResult:
    dataset_id: int
    name: str
    dataset_version: str
    case_count: int
    assertion_count: int
    deactivated_case_keys: tuple[str, ...]
    deactivated_assertions: tuple[DeactivatedAssertion, ...]
    swept_run_ids: tuple[int, ...]


class DatasetImportBlockedByRunningRunError(RuntimeError):
    code = "dataset_import_blocked_by_running_run"

    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_version: str,
        dataset_id: int,
        running_run_id: int,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.dataset_id = dataset_id
        self.running_run_id = running_run_id
        super().__init__(
            f"{self.code}: dataset {dataset_name}@{dataset_version} (id={dataset_id}) は"
            f"実行中のrun {running_run_id} があるため差し替えできません。"
            "古いrunの場合は実行状態の整理処理を行ってから再試行してください。"
        )


async def import_dataset_full_replacement(
    session: AsyncSession,
    document: Mapping[str, object],
    *,
    now: datetime | None = None,
    stale_timeout: timedelta = DEFAULT_STALE_TIMEOUT,
) -> FullReplacementImportResult:
    dataset_input = parse_dataset_import_document(document)
    repository = DatasetRepository(session)
    existing_dataset = await repository.get_by_name_and_version(
        dataset_input.name,
        dataset_input.dataset_version,
    )

    swept_run_ids: tuple[int, ...] = ()
    if existing_dataset is not None:
        sweep_result = await sweep_stale_runs_for_dataset_import(
            session,
            dataset_id=existing_dataset.dataset_id,
            now=now,
            stale_timeout=stale_timeout,
        )
        swept_run_ids = sweep_result.aborted_run_ids
        running_run_id = await _find_running_run_id(session, existing_dataset.dataset_id)
        if running_run_id is not None:
            raise DatasetImportBlockedByRunningRunError(
                dataset_name=dataset_input.name,
                dataset_version=dataset_input.dataset_version,
                dataset_id=existing_dataset.dataset_id,
                running_run_id=running_run_id,
            )

    import_result = await import_dataset_document(session, document)
    deactivated_case_keys = await _deactivate_missing_cases(
        session,
        import_result.dataset_id,
        {case.case_key for case in dataset_input.cases},
    )
    deactivated_assertions = await _deactivate_missing_assertions(
        session,
        import_result.dataset_id,
        dataset_input.cases,
    )

    return FullReplacementImportResult(
        dataset_id=import_result.dataset_id,
        name=import_result.name,
        dataset_version=import_result.dataset_version,
        case_count=import_result.case_count,
        assertion_count=import_result.assertion_count,
        deactivated_case_keys=deactivated_case_keys,
        deactivated_assertions=deactivated_assertions,
        swept_run_ids=swept_run_ids,
    )


async def _find_running_run_id(session: AsyncSession, dataset_id: int) -> int | None:
    result = await session.execute(
        select(RunRecord.run_id)
        .where(
            RunRecord.dataset_id == dataset_id,
            RunRecord.status == RunStatus.RUNNING.value,
        )
        .order_by(RunRecord.run_id)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _deactivate_missing_cases(
    session: AsyncSession,
    dataset_id: int,
    active_case_keys: set[str],
) -> tuple[str, ...]:
    result = await session.execute(
        select(
            EVALUATION_CASES.c.case_id,
            EVALUATION_CASES.c.case_key,
            EVALUATION_CASES.c.is_active,
        )
        .where(EVALUATION_CASES.c.dataset_id == dataset_id)
        .order_by(EVALUATION_CASES.c.case_key)
    )

    deactivated: list[str] = []
    for row in result.mappings():
        case_key = _str_value(row["case_key"])
        is_active = _bool_value(row["is_active"])
        if case_key in active_case_keys or not is_active:
            continue
        await session.execute(
            update(EVALUATION_CASES)
            .where(EVALUATION_CASES.c.case_id == _int_value(row["case_id"]))
            .values(is_active=False)
        )
        deactivated.append(case_key)

    await session.flush()
    return tuple(deactivated)


async def _deactivate_missing_assertions(
    session: AsyncSession,
    dataset_id: int,
    cases: tuple[DatasetCaseInput, ...],
) -> tuple[DeactivatedAssertion, ...]:
    deactivated: list[DeactivatedAssertion] = []
    for case in cases:
        case_id = await _find_case_id(session, dataset_id, case.case_key)
        assertion_ids = {assertion.id for assertion in case.assertions}
        result = await session.execute(
            select(
                ASSERTIONS.c.assertion_db_id,
                ASSERTIONS.c.id,
                ASSERTIONS.c.is_active,
            )
            .where(ASSERTIONS.c.case_id == case_id)
            .order_by(ASSERTIONS.c.id)
        )

        for row in result.mappings():
            assertion_id = _str_value(row["id"])
            is_active = _bool_value(row["is_active"])
            if assertion_id in assertion_ids or not is_active:
                continue
            await session.execute(
                update(ASSERTIONS)
                .where(ASSERTIONS.c.assertion_db_id == _int_value(row["assertion_db_id"]))
                .values(is_active=False)
            )
            deactivated.append(
                DeactivatedAssertion(case_key=case.case_key, assertion_id=assertion_id)
            )

    await session.flush()
    return tuple(deactivated)


async def _find_case_id(session: AsyncSession, dataset_id: int, case_key: str) -> int:
    result = await session.execute(
        select(EVALUATION_CASES.c.case_id).where(
            EVALUATION_CASES.c.dataset_id == dataset_id,
            EVALUATION_CASES.c.case_key == case_key,
        )
    )
    return _int_value(result.scalar_one())


def _int_value(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError("expected int value from database")
    return value


def _str_value(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("expected str value from database")
    return value


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    raise TypeError("expected bool value from database")
