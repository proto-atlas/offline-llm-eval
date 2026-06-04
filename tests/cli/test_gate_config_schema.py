import pytest
from pydantic import ValidationError

from offline_llm_eval.cli.gate_config_schema import (
    GateConfigSchema,
    validate_gate_config_schema,
)


def test_gate_config_schemaは有効な設定を受け取る() -> None:
    config = validate_gate_config_schema(
        {
            "pass_rate_min": 0.9,
            "fail_rate_max": 0.1,
            "max_pass_rate_delta": 0.05,
            "max_fail_rate_delta": 0.02,
            "max_skipped_ratio_delta": 0.03,
            "high_severity_must_pass": True,
            "fail_on_secret_leak": True,
            "max_high_severity_skipped": 0,
        }
    )

    assert config == GateConfigSchema(
        pass_rate_min=0.9,
        fail_rate_max=0.1,
        max_pass_rate_delta=0.05,
        max_fail_rate_delta=0.02,
        max_skipped_ratio_delta=0.03,
        high_severity_must_pass=True,
        fail_on_secret_leak=True,
        max_high_severity_skipped=0,
    )


def test_gate_config_schemaは未指定の閾値をNoneにする() -> None:
    config = validate_gate_config_schema({})

    assert config.model_dump() == {
        "pass_rate_min": None,
        "fail_rate_max": None,
        "max_pass_rate_delta": None,
        "max_fail_rate_delta": None,
        "max_skipped_ratio_delta": None,
        "high_severity_must_pass": False,
        "fail_on_secret_leak": False,
        "max_high_severity_skipped": None,
    }


def test_gate_config_schemaは未知キーを拒否する() -> None:
    with pytest.raises(ValidationError) as error:
        validate_gate_config_schema({"fail_on_sse_error_event": True})

    assert error.value.errors()[0]["type"] == "extra_forbidden"


def test_gate_config_schemaはrate範囲外を拒否する() -> None:
    with pytest.raises(ValidationError) as error:
        validate_gate_config_schema({"pass_rate_min": 1.01})

    assert error.value.errors()[0]["loc"] == ("pass_rate_min",)


def test_gate_config_schemaはdelta範囲外を拒否する() -> None:
    with pytest.raises(ValidationError) as error:
        validate_gate_config_schema({"max_fail_rate_delta": -0.01})

    assert error.value.errors()[0]["loc"] == ("max_fail_rate_delta",)


def test_gate_config_schemaは負のskipped件数を拒否する() -> None:
    with pytest.raises(ValidationError) as error:
        validate_gate_config_schema({"max_high_severity_skipped": -1})

    assert error.value.errors()[0]["loc"] == ("max_high_severity_skipped",)


def test_gate_config_schemaはbool文字列を拒否する() -> None:
    with pytest.raises(ValidationError) as error:
        validate_gate_config_schema({"high_severity_must_pass": "yes"})

    assert error.value.errors()[0]["type"] == "bool_type"


def test_gate_config_schemaは数値文字列を拒否する() -> None:
    with pytest.raises(ValidationError) as error:
        validate_gate_config_schema({"pass_rate_min": "0.9"})

    assert error.value.errors()[0]["type"] == "float_type"
