from offline_llm_eval.diff.classification import (
    AssertionForDiff,
    CaseForDiff,
    DiffClassification,
    DiffClassificationType,
    classify_diff,
)
from offline_llm_eval.evaluator.status import CaseResultStatus


def test_changed_added_removed_caseを分類する() -> None:
    classifications = classify_diff(
        baseline_cases=(
            CaseForDiff("case_changed", CaseResultStatus.PASS),
            CaseForDiff("case_removed", CaseResultStatus.FAILED),
        ),
        current_cases=(
            CaseForDiff("case_changed", CaseResultStatus.FAILED),
            CaseForDiff("case_added", CaseResultStatus.PASS),
        ),
    )

    assert classifications == (
        DiffClassification(DiffClassificationType.CHANGED_CASE, "case_changed"),
        DiffClassification(DiffClassificationType.ADDED_CASE, "case_added"),
        DiffClassification(DiffClassificationType.REMOVED_CASE, "case_removed"),
    )


def test_inactive_caseはremoved_caseとして分類する() -> None:
    classifications = classify_diff(
        baseline_cases=(CaseForDiff("case_removed", CaseResultStatus.PASS),),
        current_cases=(
            CaseForDiff(
                "case_removed",
                CaseResultStatus.PASS,
                is_active=False,
                assertions=(AssertionForDiff("assertion_removed", is_active=False),),
            ),
        ),
    )

    assert classifications == (
        DiffClassification(DiffClassificationType.REMOVED_CASE, "case_removed"),
    )


def test_active_case内のinactive_assertionをremoved_assertionとして分類する() -> None:
    classifications = classify_diff(
        baseline_cases=(CaseForDiff("case_active", CaseResultStatus.PASS),),
        current_cases=(
            CaseForDiff(
                "case_active",
                CaseResultStatus.PASS,
                assertions=(
                    AssertionForDiff("assertion_active"),
                    AssertionForDiff("assertion_removed", is_active=False),
                ),
            ),
        ),
    )

    assert classifications == (
        DiffClassification(
            DiffClassificationType.REMOVED_ASSERTION,
            "case_active",
            "assertion_removed",
        ),
    )


def test文字列statusもchanged_case分類に使える() -> None:
    classifications = classify_diff(
        baseline_cases=(CaseForDiff("case_changed", "pass"),),
        current_cases=(CaseForDiff("case_changed", "needs_review"),),
    )

    assert classifications == (
        DiffClassification(DiffClassificationType.CHANGED_CASE, "case_changed"),
    )
