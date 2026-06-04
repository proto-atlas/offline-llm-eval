from offline_llm_eval.evidence.sanitize import (
    sanitize_evidence_text,
    sanitize_evidence_value,
    sanitize_reviewer_note,
)


def join_parts(*parts: str) -> str:
    return "".join(parts)


def test_sanitize_evidence_textはsecret値をcategory付きmaskに置換する() -> None:
    secret_value = join_parts("AKIA", "12345678", "90ABCDEF")

    result = sanitize_evidence_text(f"token={secret_value}")

    assert result == "token=[masked:aws_access_key]"
    assert secret_value not in result


def test_sanitize_reviewer_noteはsecretをmaskする() -> None:
    secret_value = join_parts("sk-proj-", "abcdefghijklmnopqrstuvwxyz")

    result = sanitize_reviewer_note(f"review note includes {secret_value}")

    assert result == "review note includes [masked:openai_api_key]"
    assert secret_value not in result


def test_sanitize_reviewer_noteはnoneを保持する() -> None:
    assert sanitize_reviewer_note(None) is None


def test_sanitize_evidence_valueは入れ子の文字列だけsanitizeする() -> None:
    aws_key = join_parts("AKIA", "12345678", "90ABCDEF")
    bearer_token = join_parts("Bearer ", "ey", "header.payload.signature")

    result = sanitize_evidence_value(
        {
            "answer": f"unsafe {aws_key}",
            "citations": [{"snippet": f"token {bearer_token}"}],
            "metadata": {"count": 2, "ok": True, "missing": None},
        }
    )

    assert result == {
        "answer": "unsafe [masked:aws_access_key]",
        "citations": [{"snippet": "token [masked:bearer_token]"}],
        "metadata": {"count": 2, "ok": True, "missing": None},
    }
    assert aws_key not in str(result)
    assert bearer_token not in str(result)


def test_sanitize_evidence_valueはdict_keyのsecretもmaskする() -> None:
    aws_key = join_parts("AKIA", "12345678", "90ABCDEF")
    nested_aws_key = join_parts("AKIA", "22345678", "90ABCDEF")

    result = sanitize_evidence_value(
        {
            aws_key: "value",
            "nested": {nested_aws_key: "inner"},
        }
    )

    assert result == {
        "[masked:aws_access_key]": "value",
        "nested": {"[masked:aws_access_key]": "inner"},
    }
    assert aws_key not in str(result)
    assert nested_aws_key not in str(result)
