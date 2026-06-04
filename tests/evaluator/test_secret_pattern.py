import pytest

from offline_llm_eval.evaluator.secret_pattern import (
    SecretPatternCategory,
    SecretScanStatus,
    build_secret_scan_input,
    collect_string_fields,
    scan_secret_fields,
    scan_secret_text,
)


def join_parts(*parts: str) -> str:
    return "".join(parts)


@pytest.mark.parametrize(
    ("secret_text", "detail_code"),
    [
        (
            join_parts("AKIA", "12345678", "90ABCDEF"),
            SecretPatternCategory.AWS_ACCESS_KEY.value,
        ),
        (
            join_parts("-----BEGIN ", "PRIVATE KEY-----"),
            SecretPatternCategory.PEM_PRIVATE_KEY.value,
        ),
        (
            join_parts("eyJ", "header", ".payload", ".signature"),
            SecretPatternCategory.JWT.value,
        ),
        (
            join_parts("Bearer ", "ey", "header.payload.signature"),
            SecretPatternCategory.BEARER_TOKEN.value,
        ),
    ],
)
def test_base_pattern_必須4種を検出する(secret_text: str, detail_code: str) -> None:
    result = scan_secret_text(secret_text)

    assert result.status is SecretScanStatus.FAILED
    assert result.detail_code == detail_code


@pytest.mark.parametrize(
    ("secret_text", "detail_code"),
    [
        (
            join_parts("sk-proj-", "abcdefghijklmnopqrstuvwxyz"),
            SecretPatternCategory.OPENAI_API_KEY.value,
        ),
        (
            join_parts("sk-ant-api03-", "abcdefghijklmnopqrstuvwxyz"),
            SecretPatternCategory.ANTHROPIC_API_KEY.value,
        ),
        (
            join_parts("AIza", "1234567890abcdefghijklmnopqrstuvwxy"),
            SecretPatternCategory.GOOGLE_API_KEY.value,
        ),
        (
            join_parts("github_pat_", "abcdefghijklmnopqrstuvwxyz"),
            SecretPatternCategory.GITHUB_PAT.value,
        ),
        (
            join_parts("xoxb-", "1234567890-ABCDEFGHIJ"),
            SecretPatternCategory.SLACK_TOKEN.value,
        ),
        (
            join_parts("sk_live_", "1234567890abcdef"),
            SecretPatternCategory.STRIPE_KEY.value,
        ),
        (
            join_parts("-----BEGIN ", "RSA PRIVATE KEY-----"),
            SecretPatternCategory.GENERIC_PRIVATE_KEY.value,
        ),
    ],
)
def test_extension_pattern_拡張カテゴリを検出する(secret_text: str, detail_code: str) -> None:
    result = scan_secret_text(secret_text)

    assert result.status is SecretScanStatus.FAILED
    assert result.detail_code == detail_code


def test_scan_secret_text_secretがなければpassになる() -> None:
    result = scan_secret_text("normal answer with citations and metadata")

    assert result.status is SecretScanStatus.PASS
    assert result.detail_code is None


def test_scan_secret_text_secret値そのものは結果に保持しない() -> None:
    secret_text = join_parts("AKIA", "12345678", "90ABCDEF")

    result = scan_secret_text(secret_text)

    assert result.status is SecretScanStatus.FAILED
    assert result.detail_code == "aws_access_key"
    assert not hasattr(result, "matched_value")
    assert not hasattr(result, "secret_value")


def test_collect_string_fields_入れ子の全stringを順序通り集める() -> None:
    result = collect_string_fields(
        {
            "answer": "answer text",
            "citations": [{"source": "docs/source", "snippet": "snippet text"}],
            "metadata": {"reason": "nested text", "count": 3},
        }
    )

    assert result == (
        "answer text",
        "docs/source",
        "snippet text",
        "nested text",
    )


def test_build_secret_scan_input_改行delimiterで連結する() -> None:
    result = build_secret_scan_input(
        {
            "answer": "first",
            "metadata": {"reason": "second"},
        }
    )

    assert result == "first\nsecond"


def test_build_secret_scan_input_隣接field境界で偶発secretを作らない() -> None:
    scan_input = build_secret_scan_input(
        {
            "left": join_parts("AKIA", "12345678"),
            "right": "90ABCDEF",
        }
    )

    result = scan_secret_text(scan_input)

    assert scan_input == "AKIA12345678\n90ABCDEF"
    assert result.status is SecretScanStatus.PASS


def test_scan_secret_fields_response全体からsecretを検出する() -> None:
    result = scan_secret_fields(
        {
            "answer": "answer text",
            "citations": [{"source": "docs/source", "snippet": "safe snippet"}],
            "metadata": {"token": join_parts("AKIA", "12345678", "90ABCDEF")},
        }
    )

    assert result.status is SecretScanStatus.FAILED
    assert result.detail_code == "aws_access_key"


def test_scan_secret_fields_bytesは文字列として扱わない() -> None:
    result = scan_secret_fields(
        {
            "answer": join_parts("AKIA", "12345678", "90ABCDEF").encode(),
        }
    )

    assert result.status is SecretScanStatus.PASS
