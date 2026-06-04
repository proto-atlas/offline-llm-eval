from pathlib import Path

import pytest
from pydantic import ValidationError

from offline_llm_eval.cli.gate_config import load_gate_config, load_gate_config_file
from offline_llm_eval.cli.gate_config_schema import GateConfigSchema
from offline_llm_eval.util.document import DocumentLoadError


def test_load_gate_config_fileはyamlをschemaへ変換する(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text("pass_rate_min: 0.9\nfail_on_secret_leak: true\n", encoding="utf-8")

    assert load_gate_config_file(path) == GateConfigSchema(
        pass_rate_min=0.9,
        fail_on_secret_leak=True,
    )


def test_load_gate_configはcli引数でyamlを上書きする(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text(
        "pass_rate_min: 0.8\nfail_on_secret_leak: true\n",
        encoding="utf-8",
    )

    config = load_gate_config(
        config_path=path,
        cli_overrides={
            "pass_rate_min": 0.95,
            "fail_on_secret_leak": False,
            "max_high_severity_skipped": 1,
        },
    )

    assert config == GateConfigSchema(
        pass_rate_min=0.95,
        fail_on_secret_leak=False,
        max_high_severity_skipped=1,
    )


def test_load_gate_configはNoneのcli引数を無視する(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text("pass_rate_min: 0.8\n", encoding="utf-8")

    config = load_gate_config(
        config_path=path,
        cli_overrides={"pass_rate_min": None},
    )

    assert config.pass_rate_min == 0.8


def test_load_gate_configはconfigなしならcli引数だけを使う() -> None:
    config = load_gate_config(
        cli_overrides={
            "fail_rate_max": 0.2,
            "high_severity_must_pass": True,
        }
    )

    assert config == GateConfigSchema(
        fail_rate_max=0.2,
        high_severity_must_pass=True,
    )


def test_load_gate_configはcli未知キーを拒否する() -> None:
    with pytest.raises(ValidationError) as error:
        load_gate_config(cli_overrides={"fail_on_sse_error_event": True})

    assert error.value.errors()[0]["type"] == "extra_forbidden"


def test_load_gate_config_fileはschema違反を拒否する(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text("pass_rate_min: 1.1\n", encoding="utf-8")

    with pytest.raises(ValidationError) as error:
        load_gate_config_file(path)

    assert error.value.errors()[0]["loc"] == ("pass_rate_min",)


def test_load_gate_config_fileは空yamlをvalidation_errorにする(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text(" ", encoding="utf-8")

    with pytest.raises(DocumentLoadError) as error:
        load_gate_config_file(path)

    assert str(error.value) == f"validation_error: {path} document は空にできません。"
