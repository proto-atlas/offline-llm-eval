import json
import logging
from datetime import datetime

import pytest
from sqlalchemy import text

from offline_llm_eval.db.engine import create_async_db_engine
from offline_llm_eval.evaluator.secret_pattern import SECRET_SCAN_ASSERTION_ID
from offline_llm_eval.evidence.markdown import (
    EvidenceAssertion,
    EvidenceCase,
    EvidenceReport,
    EvidenceRunMetadata,
    render_evidence_markdown,
)
from offline_llm_eval.logging.structured import JsonLineFormatter
from offline_llm_eval.run.metrics import calculate_run_metrics
from offline_llm_eval.security.secret_provider import EnvSecretProvider, SecretNotFoundError

PROVIDER_SECRET_VALUE = "provider-secret-value"
AWS_ACCESS_KEY_VALUE = "".join(("AKIA", "12345678", "90ABCDEF"))


def test_get_returns_secret_value_through_explicit_reveal() -> None:
    secret = EnvSecretProvider({"PROVIDER_API_KEY": PROVIDER_SECRET_VALUE}).get("PROVIDER_API_KEY")

    assert secret.name == "PROVIDER_API_KEY"
    assert secret.reveal() == PROVIDER_SECRET_VALUE


def test_secret_value_string_representations_are_redacted() -> None:
    secret = EnvSecretProvider({"PROVIDER_API_KEY": PROVIDER_SECRET_VALUE}).get("PROVIDER_API_KEY")

    assert str(secret) == "********"
    assert repr(secret) == "SecretValue(name='PROVIDER_API_KEY', value='********')"
    assert PROVIDER_SECRET_VALUE not in str(secret)
    assert PROVIDER_SECRET_VALUE not in repr(secret)


def test_missing_secret_error_does_not_include_other_env_values() -> None:
    provider = EnvSecretProvider({"PROVIDER_API_KEY": PROVIDER_SECRET_VALUE})

    with pytest.raises(SecretNotFoundError) as error:
        provider.get("MISSING_API_KEY")

    assert str(error.value) == "Secret が設定されていません: MISSING_API_KEY"
    assert PROVIDER_SECRET_VALUE not in str(error.value)


def test_blank_secret_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Secret name は空にできません。"):
        EnvSecretProvider({"PROVIDER_API_KEY": PROVIDER_SECRET_VALUE}).get(" ")


def test_blank_secret_value_is_treated_as_missing() -> None:
    with pytest.raises(SecretNotFoundError) as error:
        EnvSecretProvider({"PROVIDER_API_KEY": ""}).get("PROVIDER_API_KEY")

    assert str(error.value) == "Secret が設定されていません: PROVIDER_API_KEY"


def test_secret_value_is_masked_in_structured_log_output() -> None:
    secret = EnvSecretProvider({"PROVIDER_API_KEY": PROVIDER_SECRET_VALUE}).get("PROVIDER_API_KEY")
    record = logging.LogRecord(
        name="offline_llm_eval.security",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="provider key=%s",
        args=(secret.reveal(),),
        exc_info=None,
    )
    record.created = 0.0
    record.structured = {"provider": {"api_key": secret.reveal()}}

    output = JsonLineFormatter(secret_values=[secret.reveal()]).format(record)

    assert PROVIDER_SECRET_VALUE not in output
    assert json.loads(output)["message"] == "provider key=********"


@pytest.mark.asyncio
async def test_secret_value_string_representation_does_not_expose_db_value() -> None:
    secret = EnvSecretProvider({"PROVIDER_API_KEY": PROVIDER_SECRET_VALUE}).get("PROVIDER_API_KEY")
    engine = create_async_db_engine("sqlite+aiosqlite:///:memory:")
    stored_payload = json.dumps(
        {"name": secret.name, "value": str(secret), "debug": repr(secret)},
        ensure_ascii=False,
    )

    try:
        async with engine.begin() as connection:
            await connection.execute(text("create table secret_audit (payload text not null)"))
            await connection.execute(
                text("insert into secret_audit (payload) values (:payload)"),
                {"payload": stored_payload},
            )
            result = await connection.execute(text("select payload from secret_audit"))
    finally:
        await engine.dispose()

    stored_text = str(result.scalar_one())
    assert PROVIDER_SECRET_VALUE not in stored_text
    assert "********" in stored_text


def test_secret_like_evidence_values_are_masked_in_markdown() -> None:
    secret = EnvSecretProvider({"PROVIDER_API_KEY": AWS_ACCESS_KEY_VALUE}).get("PROVIDER_API_KEY")
    report = EvidenceReport(
        run=EvidenceRunMetadata(
            run_id=1,
            dataset_name=f"dataset {secret.reveal()}",
            dataset_version="1.0.0",
            target_label="local",
            target_version=None,
            status="completed",
            started_at=datetime(2026, 5, 27, 18, 0, 0),
            completed_at=None,
        ),
        metrics=calculate_run_metrics(("failed",)),
        cases=(
            EvidenceCase(
                case_key="case_secret",
                status="failed",
                final_status=None,
                reviewer_note=f"review note {secret.reveal()}",
                assertions=(
                    EvidenceAssertion(
                        assertion_id=SECRET_SCAN_ASSERTION_ID,
                        assertion_type="secret_scan",
                        status="failed",
                        detail=f"detected {secret.reveal()}",
                        required=True,
                        severity="high",
                        on_fail="fail",
                    ),
                ),
            ),
        ),
    )

    markdown = render_evidence_markdown(report)

    assert AWS_ACCESS_KEY_VALUE not in markdown
    assert "[masked:aws_access_key]" in markdown
