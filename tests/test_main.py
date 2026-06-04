import pytest
import uvicorn
from fastapi.testclient import TestClient

from offline_llm_eval import __version__, main
from offline_llm_eval.main import app, create_app


class UvicornRunCall:
    def __init__(self, app_path: str, host: str, port: int) -> None:
        self.app_path = app_path
        self.host = host
        self.port = port


def test_create_app_sets_openapi_metadata() -> None:
    client = TestClient(create_app())

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"] == {
        "title": "Offline LLM Eval",
        "version": __version__,
    }


def test_module_app_is_fastapi_application() -> None:
    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200


def test_run_api_server_binds_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[UvicornRunCall] = []

    def fake_run(app_path: str, *, host: str, port: int) -> None:
        calls.append(UvicornRunCall(app_path, host, port))

    monkeypatch.setattr(uvicorn, "run", fake_run)

    main.run_api_server()

    assert len(calls) == 1
    assert calls[0].app_path == "offline_llm_eval.main:app"
    assert calls[0].host == "127.0.0.1"
    assert calls[0].port == 8000
