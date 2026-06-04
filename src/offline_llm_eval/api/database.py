from typing import cast

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from offline_llm_eval.db import create_async_db_engine, create_session_factory

DB_ENGINE_STATE_KEY = "offline_llm_eval_db_engine"
DB_SESSION_FACTORY_STATE_KEY = "offline_llm_eval_db_session_factory"


def get_or_create_session_factory(app: FastAPI) -> async_sessionmaker[AsyncSession]:
    session_factory = cast(
        async_sessionmaker[AsyncSession] | None,
        getattr(app.state, DB_SESSION_FACTORY_STATE_KEY, None),
    )
    if session_factory is not None:
        return session_factory

    engine = create_async_db_engine()
    session_factory = create_session_factory(engine)
    setattr(app.state, DB_ENGINE_STATE_KEY, engine)
    setattr(app.state, DB_SESSION_FACTORY_STATE_KEY, session_factory)
    return session_factory


async def dispose_database_engine(app: FastAPI) -> None:
    engine = getattr(app.state, DB_ENGINE_STATE_KEY, None)
    if not isinstance(engine, AsyncEngine):
        return

    await engine.dispose()
    setattr(app.state, DB_ENGINE_STATE_KEY, None)
    setattr(app.state, DB_SESSION_FACTORY_STATE_KEY, None)
