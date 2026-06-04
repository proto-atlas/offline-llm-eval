import tomllib
from importlib.metadata import version
from pathlib import Path

from offline_llm_eval import __version__


def test_package_version_matches_metadata() -> None:
    assert version("offline-llm-eval") == __version__


def test_package_metadataはgate_cli_entrypointを公開する() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["offline-llm-eval-check"] == (
        "offline_llm_eval.cli.check:main"
    )
