from offline_llm_eval.dataset.assertion_model import (
    AssertionOnFail,
    AssertionSeverity,
    AssertionType,
)
from offline_llm_eval.evaluator.assertions import (
    AssertionEvaluationStatus,
    AssertionForEvaluation,
    CitationForEvaluation,
    EvaluatedAssertionResult,
    JsonValue,
    ResponseForEvaluation,
    evaluate_assertion,
    evaluate_assertions,
)


def make_assertion(
    assertion_type: AssertionType,
    expected: JsonValue,
    *,
    required: bool = True,
    on_fail: AssertionOnFail = AssertionOnFail.FAIL,
) -> AssertionForEvaluation:
    return AssertionForEvaluation(
        assertion_db_id=1,
        assertion_id="answer_check",
        assertion_type=assertion_type,
        expected=expected,
        required=required,
        severity=AssertionSeverity.HIGH,
        on_fail=on_fail,
    )


def test_exact_match_正規化後に一致したらpassになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.EXACT_MATCH, "Expected Answer"),
        ResponseForEvaluation(answer="  expected   answer  "),
    )

    assert result.status is AssertionEvaluationStatus.PASS


def test_exact_match_一致しなければfailedになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.EXACT_MATCH, "Expected Answer"),
        ResponseForEvaluation(answer="different"),
    )

    assert result.status is AssertionEvaluationStatus.FAILED
    assert result.detail == "exact_match_mismatch"


def test_normalized_contains_正規化後に含まれたらpassになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.NORMALIZED_CONTAINS, "run summary"),
        ResponseForEvaluation(answer="The RUN    SUMMARY includes counts."),
    )

    assert result.status is AssertionEvaluationStatus.PASS


def test_keyword_all_全keywordを含めばpassになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.KEYWORD_ALL, ["name", "dataset_version"]),
        ResponseForEvaluation(answer="Dataset YAML has name and dataset_version."),
    )

    assert result.status is AssertionEvaluationStatus.PASS


def test_keyword_all_欠けたkeywordがあればfailedになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.KEYWORD_ALL, ["name", "dataset_version"]),
        ResponseForEvaluation(answer="Dataset YAML has name."),
    )

    assert result.status is AssertionEvaluationStatus.FAILED
    assert result.detail == "keyword_all_missing"
    assert result.matched_value == ["dataset_version"]


def test_keyword_any_一つでも含めばpassになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.KEYWORD_ANY, ["metadata", "reviewer"]),
        ResponseForEvaluation(answer="The response includes reviewer notes."),
    )

    assert result.status is AssertionEvaluationStatus.PASS
    assert result.matched_value == "reviewer"


def test_keyword_any_どれも含まなければfailedになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.KEYWORD_ANY, ["metadata", "reviewer"]),
        ResponseForEvaluation(answer="The response includes counts."),
    )

    assert result.status is AssertionEvaluationStatus.FAILED
    assert result.detail == "keyword_any_missing"


def test_regex_一致した部分があればpassになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.REGEX, r"run-\d+"),
        ResponseForEvaluation(answer="Created run-123."),
    )

    assert result.status is AssertionEvaluationStatus.PASS
    assert result.matched_value == "run-123"


def test_regex_patternが不正ならskippedになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.REGEX, "["),
        ResponseForEvaluation(answer="Created run-123."),
    )

    assert result.status is AssertionEvaluationStatus.SKIPPED
    assert result.detail == "invalid_expected"


def test_no_answer_expected_空文字ならpassになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.NO_ANSWER_EXPECTED, True),
        ResponseForEvaluation(answer=""),
    )

    assert result.status is AssertionEvaluationStatus.PASS


def test_no_answer_expected_回答があればfailedになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.NO_ANSWER_EXPECTED, True),
        ResponseForEvaluation(answer="The token is API_TOKEN_PLACEHOLDER."),
    )

    assert result.status is AssertionEvaluationStatus.FAILED
    assert result.detail == "answer_present"


def test_citation_presence_citationがあればpassになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.CITATION_PRESENCE, True),
        ResponseForEvaluation(
            answer="See the docs.",
            citations=(CitationForEvaluation(source="docs/api-endpoints"),),
        ),
    )

    assert result.status is AssertionEvaluationStatus.PASS


def test_citation_presence_期待に反したらfailedになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.CITATION_PRESENCE, True),
        ResponseForEvaluation(answer="See the docs."),
    )

    assert result.status is AssertionEvaluationStatus.FAILED
    assert result.detail == "citation_presence_mismatch"


def test_source_id_exact_set_順序が違っても集合が一致したらpassになる() -> None:
    result = evaluate_assertion(
        make_assertion(
            AssertionType.SOURCE_ID_EXACT_SET,
            ["docs/nfr-005", "docs/api-endpoints"],
        ),
        ResponseForEvaluation(
            answer="See the docs.",
            citations=(
                CitationForEvaluation(source="docs/api-endpoints"),
                CitationForEvaluation(source="docs/nfr-005"),
            ),
        ),
    )

    assert result.status is AssertionEvaluationStatus.PASS


def test_source_id_exact_set_集合が違えばfailedになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.SOURCE_ID_EXACT_SET, ["docs/nfr-005"]),
        ResponseForEvaluation(
            answer="See the docs.",
            citations=(CitationForEvaluation(source="docs/api-endpoints"),),
        ),
    )

    assert result.status is AssertionEvaluationStatus.FAILED
    assert result.detail == "source_id_exact_set_mismatch"


def test_source_id_subset_期待sourceが実sourceに含まれればpassになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.SOURCE_ID_SUBSET, ["docs/storage-backend"]),
        ResponseForEvaluation(
            answer="See the docs.",
            citations=(
                CitationForEvaluation(source="docs/storage-backend"),
                CitationForEvaluation(source="docs/nfr-007"),
            ),
        ),
    )

    assert result.status is AssertionEvaluationStatus.PASS


def test_json_schema_schemaに合えばpassになる() -> None:
    result = evaluate_assertion(
        make_assertion(
            AssertionType.JSON_SCHEMA,
            {
                "type": "object",
                "required": ["verdict", "checked_cases"],
                "properties": {
                    "verdict": {"type": "string"},
                    "checked_cases": {"type": "integer"},
                },
            },
        ),
        ResponseForEvaluation(answer='{"verdict": "pass", "checked_cases": 3}'),
    )

    assert result.status is AssertionEvaluationStatus.PASS


def test_json_schema_requiredが欠けたらfailedになる() -> None:
    result = evaluate_assertion(
        make_assertion(
            AssertionType.JSON_SCHEMA,
            {"type": "object", "required": ["verdict"]},
        ),
        ResponseForEvaluation(answer='{"status": "pass"}'),
    )

    assert result.status is AssertionEvaluationStatus.FAILED
    assert result.detail == "json_schema_required_missing"


def test_json_schema_array_itemsが合わなければfailedになる() -> None:
    result = evaluate_assertion(
        make_assertion(
            AssertionType.JSON_SCHEMA,
            {"type": "array", "items": {"type": "integer"}},
        ),
        ResponseForEvaluation(answer='[1, "two"]'),
    )

    assert result.status is AssertionEvaluationStatus.FAILED
    assert result.detail == "json_schema_type_mismatch"


def test_json_schema_answerがjsonでなければskippedになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.JSON_SCHEMA, {"type": "object"}),
        ResponseForEvaluation(answer="not-json"),
    )

    assert result.status is AssertionEvaluationStatus.SKIPPED
    assert result.detail == "json_parse_error"


def test_json_schema_schemaが不正ならskippedになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.JSON_SCHEMA, {"type": "unknown"}),
        ResponseForEvaluation(answer='{"status": "pass"}'),
    )

    assert result.status is AssertionEvaluationStatus.SKIPPED
    assert result.detail == "invalid_expected"


def test_forbidden_phrase_禁止語がなければpassになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.FORBIDDEN_PHRASE, ["Traceback", "Exception:"]),
        ResponseForEvaluation(answer="Structured response generated."),
    )

    assert result.status is AssertionEvaluationStatus.PASS


def test_forbidden_phrase_空listならskippedになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.FORBIDDEN_PHRASE, []),
        ResponseForEvaluation(answer="Structured response generated."),
    )

    assert result.status is AssertionEvaluationStatus.SKIPPED
    assert result.detail == "invalid_expected"


def test_forbidden_phrase_禁止語があればfailedになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.FORBIDDEN_PHRASE, ["Traceback", "Exception:"]),
        ResponseForEvaluation(answer="Traceback: failed."),
    )

    assert result.status is AssertionEvaluationStatus.FAILED
    assert result.matched_value == "Traceback"


def test_latency_threshold_閾値以内ならpassになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.LATENCY_THRESHOLD, 1500),
        ResponseForEvaluation(answer="{}", latency_ms=1499),
    )

    assert result.status is AssertionEvaluationStatus.PASS


def test_latency_threshold_latencyがなければskippedになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.LATENCY_THRESHOLD, 1500),
        ResponseForEvaluation(answer="{}"),
    )

    assert result.status is AssertionEvaluationStatus.SKIPPED
    assert result.detail == "latency_missing"


def test_required_falseで失敗したらwarningになる() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.LATENCY_THRESHOLD, 1500, required=False),
        ResponseForEvaluation(answer="{}", latency_ms=2000),
    )

    assert result.status is AssertionEvaluationStatus.WARNING


def test_on_fail_warnで失敗したらwarningになる() -> None:
    result = evaluate_assertion(
        make_assertion(
            AssertionType.EXACT_MATCH,
            "expected",
            on_fail=AssertionOnFail.WARN,
        ),
        ResponseForEvaluation(answer="actual"),
    )

    assert result.status is AssertionEvaluationStatus.WARNING


def test_on_fail_needs_reviewで失敗したらfailedのままになる() -> None:
    result = evaluate_assertion(
        make_assertion(
            AssertionType.EXACT_MATCH,
            "expected",
            on_fail=AssertionOnFail.NEEDS_REVIEW,
        ),
        ResponseForEvaluation(answer="actual"),
    )

    assert result.status is AssertionEvaluationStatus.FAILED
    assert result.on_fail is AssertionOnFail.NEEDS_REVIEW


def test_evaluate_assertions_複数assertionを順序通り評価する() -> None:
    results = evaluate_assertions(
        (
            make_assertion(AssertionType.EXACT_MATCH, "expected"),
            make_assertion(AssertionType.FORBIDDEN_PHRASE, ["Traceback"]),
        ),
        ResponseForEvaluation(answer="expected"),
    )

    assert [result.status for result in results] == [
        AssertionEvaluationStatus.PASS,
        AssertionEvaluationStatus.PASS,
    ]


def test_to_db_payload_9fieldを保持する() -> None:
    result = evaluate_assertion(
        make_assertion(AssertionType.EXACT_MATCH, "expected"),
        ResponseForEvaluation(answer="actual"),
    )

    assert result.to_db_payload() == {
        "assertion_db_id": 1,
        "assertion_id": "answer_check",
        "assertion_type": AssertionType.EXACT_MATCH,
        "status": AssertionEvaluationStatus.FAILED,
        "detail": "exact_match_mismatch",
        "matched_value": "actual",
        "expected": "expected",
        "required": True,
        "severity": AssertionSeverity.HIGH,
        "on_fail": AssertionOnFail.FAIL,
    }


def test_to_db_payloadはsecret形状の値をmaskする() -> None:
    expected_secret = "".join(("AKIA", "12345678", "90ABCDEF"))
    answer_secret = "".join(("AKIA", "22345678", "90ABCDEF"))
    result = evaluate_assertion(
        make_assertion(AssertionType.EXACT_MATCH, expected_secret),
        ResponseForEvaluation(answer=answer_secret),
    )

    assert result.to_db_payload()["matched_value"] == "[masked:aws_access_key]"
    assert result.to_db_payload()["expected"] == "[masked:aws_access_key]"


def test_to_db_payloadはdetailのsecret形状値をmaskする() -> None:
    detail_secret = "".join(("AKIA", "32345678", "90ABCDEF"))
    result = EvaluatedAssertionResult(
        assertion_db_id=1,
        assertion_id="answer_check",
        assertion_type=AssertionType.EXACT_MATCH,
        status=AssertionEvaluationStatus.FAILED,
        detail=f"detail {detail_secret}",
        matched_value=None,
        expected=None,
        required=True,
        severity=AssertionSeverity.HIGH,
        on_fail=AssertionOnFail.FAIL,
    )

    assert result.to_db_payload()["detail"] == "detail [masked:aws_access_key]"
