from datetime import datetime
from typing import Final

from sqlalchemy.exc import IntegrityError

from offline_llm_eval.dataset.repository import JsonObject
from offline_llm_eval.run.repository import RunRepository, RunSnapshot

RUNNING_RUN_INDEX: Final = "uq_runs_running_dataset_target"
CONCURRENT_RUN_BLOCKED_CODE: Final = "concurrent_run_blocked"
CONCURRENT_RUN_BLOCKED_EXIT_CODE: Final = 2

_SQLITE_UNIQUE_CONSTRAINT_TEXT: Final = "unique constraint failed"
_RUNS_DATASET_ID_COLUMN: Final = "runs.dataset_id"
_RUNS_TARGET_LABEL_COLUMN: Final = "runs.target_label"


class ConcurrentRunBlockedError(RuntimeError):
    code = CONCURRENT_RUN_BLOCKED_CODE
    exit_code = CONCURRENT_RUN_BLOCKED_EXIT_CODE

    def __init__(self, *, dataset_id: int, target_label: str) -> None:
        self.dataset_id = dataset_id
        self.target_label = target_label
        super().__init__(f"{self.code}: dataset_id={dataset_id} target_label={target_label}")


async def create_run_with_concurrency_guard(
    repository: RunRepository,
    *,
    dataset_id: int,
    target_label: str,
    target_version: str | None = None,
    gate_config_snapshot_json: JsonObject | None = None,
    started_at: datetime | None = None,
) -> RunSnapshot:
    try:
        return await repository.create_run(
            dataset_id=dataset_id,
            target_label=target_label,
            target_version=target_version,
            gate_config_snapshot_json=gate_config_snapshot_json,
            started_at=started_at,
        )
    except IntegrityError as error:
        if is_running_run_integrity_error(error):
            raise ConcurrentRunBlockedError(
                dataset_id=dataset_id,
                target_label=target_label,
            ) from error
        raise


def is_running_run_integrity_error(error: IntegrityError) -> bool:
    normalized = _integrity_error_text(error).casefold()
    if RUNNING_RUN_INDEX in normalized:
        return True

    return (
        _SQLITE_UNIQUE_CONSTRAINT_TEXT in normalized
        and _RUNS_DATASET_ID_COLUMN in normalized
        and _RUNS_TARGET_LABEL_COLUMN in normalized
    )


def _integrity_error_text(error: IntegrityError) -> str:
    parts: tuple[object, ...] = (error.orig, error.statement, error.params)
    return " ".join(str(part) for part in parts if part is not None)
