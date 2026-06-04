from collections.abc import Mapping
from pathlib import Path

from offline_llm_eval.cli.gate_config_schema import (
    GateConfigSchema,
    validate_gate_config_schema,
)
from offline_llm_eval.util.yaml_loader import load_yaml_file


def load_gate_config_file(path: Path) -> GateConfigSchema:
    return validate_gate_config_schema(load_yaml_file(path))


def load_gate_config(
    *,
    config_path: Path | None = None,
    cli_overrides: Mapping[str, object] | None = None,
) -> GateConfigSchema:
    config_data: dict[str, object] = {}
    if config_path is not None:
        config_data.update(load_yaml_file(config_path))

    if cli_overrides is not None:
        config_data.update(_drop_unspecified_cli_values(cli_overrides))

    return validate_gate_config_schema(config_data)


def _drop_unspecified_cli_values(values: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in values.items() if value is not None}
