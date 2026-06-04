from collections.abc import Mapping
from os import environ

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DATABASE_URL_ENV = "DATABASE_URL"
DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./offline_llm_eval.db"


def get_database_url(env: Mapping[str, str] | None = None) -> str:
    values = environ if env is None else env
    database_url = values.get(DATABASE_URL_ENV)
    if database_url is None or database_url.strip() == "":
        return DEFAULT_DATABASE_URL

    return database_url


def create_async_db_engine(database_url: str | None = None, *, echo: bool = False) -> AsyncEngine:
    return create_async_engine(database_url or get_database_url(), echo=echo)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
