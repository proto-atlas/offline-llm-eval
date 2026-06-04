import pytest
from sqlalchemy import text

from offline_llm_eval.db.engine import (
    DATABASE_URL_ENV,
    DEFAULT_DATABASE_URL,
    create_async_db_engine,
    create_session_factory,
    get_database_url,
)

POSTGRESQL_DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/dbname"


def test_get_database_url_returns_default_when_env_is_missing() -> None:
    assert get_database_url({}) == DEFAULT_DATABASE_URL


def test_get_database_url_returns_default_when_env_is_blank() -> None:
    assert get_database_url({DATABASE_URL_ENV: " "}) == DEFAULT_DATABASE_URL


def test_get_database_url_uses_env_override() -> None:
    assert get_database_url({DATABASE_URL_ENV: POSTGRESQL_DATABASE_URL}) == POSTGRESQL_DATABASE_URL


@pytest.mark.asyncio
async def test_create_async_db_engine_uses_default_sqlite_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)
    engine = create_async_db_engine()
    try:
        assert engine.url.drivername == "sqlite+aiosqlite"
        assert engine.url.database == "./offline_llm_eval.db"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_create_async_db_engine_uses_postgresql_asyncpg_url() -> None:
    engine = create_async_db_engine(POSTGRESQL_DATABASE_URL)
    try:
        assert engine.url.drivername == "postgresql+asyncpg"
        assert engine.url.username == "user"
        assert engine.url.host == "localhost"
        assert engine.url.database == "dbname"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_async_engine_executes_sqlite_memory_query() -> None:
    engine = create_async_db_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.connect() as connection:
            result = await connection.execute(text("select 1"))

        assert result.scalar_one() == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_session_factory_executes_sqlite_memory_query() -> None:
    engine = create_async_db_engine("sqlite+aiosqlite:///:memory:")
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            result = await session.execute(text("select 1"))

        assert result.scalar_one() == 1
    finally:
        await engine.dispose()
