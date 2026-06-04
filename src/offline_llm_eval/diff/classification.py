from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from offline_llm_eval.evaluator.status import CaseResultStatus

type CaseStatusValue = CaseResultStatus | str


class DiffClassificationType(StrEnum):
    CHANGED_CASE = "changed_case"
    ADDED_CASE = "added_case"
    REMOVED_CASE = "removed_case"
    REMOVED_ASSERTION = "removed_assertion"


@dataclass(frozen=True, slots=True)
class AssertionForDiff:
    assertion_id: str
    is_active: bool = True


@dataclass(frozen=True, slots=True)
class CaseForDiff:
    case_key: str
    status: CaseStatusValue | None = None
    is_active: bool = True
    assertions: tuple[AssertionForDiff, ...] = ()


@dataclass(frozen=True, slots=True)
class DiffClassification:
    classification: DiffClassificationType
    case_key: str
    assertion_id: str | None = None


def classify_diff(
    *,
    baseline_cases: Sequence[CaseForDiff],
    current_cases: Sequence[CaseForDiff],
) -> tuple[DiffClassification, ...]:
    baseline_by_key = _cases_by_key(baseline_cases)
    current_by_key = _cases_by_key(current_cases)
    classifications: list[DiffClassification] = []

    classifications.extend(_changed_cases(baseline_by_key, current_by_key))
    classifications.extend(_added_cases(baseline_by_key, current_by_key))
    classifications.extend(_removed_cases(baseline_by_key, current_by_key))
    classifications.extend(_removed_assertions(current_by_key))

    return tuple(classifications)


def _cases_by_key(cases: Sequence[CaseForDiff]) -> dict[str, CaseForDiff]:
    return {case.case_key: case for case in cases}


def _changed_cases(
    baseline_by_key: dict[str, CaseForDiff],
    current_by_key: dict[str, CaseForDiff],
) -> tuple[DiffClassification, ...]:
    classifications: list[DiffClassification] = []
    shared_keys = sorted(baseline_by_key.keys() & current_by_key.keys())
    for case_key in shared_keys:
        baseline_case = baseline_by_key[case_key]
        current_case = current_by_key[case_key]
        if not baseline_case.is_active or not current_case.is_active:
            continue
        if _case_status(baseline_case) == _case_status(current_case):
            continue
        classifications.append(DiffClassification(DiffClassificationType.CHANGED_CASE, case_key))

    return tuple(classifications)


def _added_cases(
    baseline_by_key: dict[str, CaseForDiff],
    current_by_key: dict[str, CaseForDiff],
) -> tuple[DiffClassification, ...]:
    added_keys = sorted(current_by_key.keys() - baseline_by_key.keys())
    return tuple(
        DiffClassification(DiffClassificationType.ADDED_CASE, case_key)
        for case_key in added_keys
        if current_by_key[case_key].is_active
    )


def _removed_cases(
    baseline_by_key: dict[str, CaseForDiff],
    current_by_key: dict[str, CaseForDiff],
) -> tuple[DiffClassification, ...]:
    removed_keys = set(baseline_by_key.keys() - current_by_key.keys())
    removed_keys.update(case.case_key for case in current_by_key.values() if not case.is_active)
    return tuple(
        DiffClassification(DiffClassificationType.REMOVED_CASE, case_key)
        for case_key in sorted(removed_keys)
    )


def _removed_assertions(
    current_by_key: dict[str, CaseForDiff],
) -> tuple[DiffClassification, ...]:
    classifications: list[DiffClassification] = []
    for case_key, case in sorted(current_by_key.items()):
        if not case.is_active:
            continue
        for assertion in sorted(case.assertions, key=lambda item: item.assertion_id):
            if assertion.is_active:
                continue
            classifications.append(
                DiffClassification(
                    DiffClassificationType.REMOVED_ASSERTION,
                    case_key,
                    assertion.assertion_id,
                )
            )

    return tuple(classifications)


def _case_status(case: CaseForDiff) -> CaseResultStatus | None:
    if case.status is None:
        return None
    return CaseResultStatus(case.status)
