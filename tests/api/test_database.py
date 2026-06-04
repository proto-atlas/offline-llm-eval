import pytest
from fastapi import FastAPI

from offline_llm_eval.api.database import (
    DB_ENGINE_STATE_KEY,
    DB_SESSION_FACTORY_STATE_KEY,
    dispose_database_engine,
    get_or_create_session_factory,
)


@pytest.mark.asyncio
async def test_get_or_create_session_factoryはapp内で同じfactoryを再利用する() -> None:
    app = FastAPI()

    first = get_or_create_session_factory(app)
    second = get_or_create_session_factory(app)

    assert first is second
    await dispose_database_engine(app)


@pytest.mark.asyncio
async def test_dispose_database_engineはstateを初期化する() -> None:
    app = FastAPI()
    get_or_create_session_factory(app)

    await dispose_database_engine(app)

    assert getattr(app.state, DB_ENGINE_STATE_KEY) is None
    assert getattr(app.state, DB_SESSION_FACTORY_STATE_KEY) is None
