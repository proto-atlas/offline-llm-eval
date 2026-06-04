from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from offline_llm_eval import __version__
from offline_llm_eval.api.cases import router as cases_router
from offline_llm_eval.api.database import (
    dispose_database_engine,
    get_or_create_session_factory,
)
from offline_llm_eval.api.diff import router as diff_router
from offline_llm_eval.api.evidence import router as evidence_router
from offline_llm_eval.api.health import router as health_router
from offline_llm_eval.api.review import router as review_router
from offline_llm_eval.api.runs import router as runs_router
from offline_llm_eval.api.validation_error import request_validation_error_handler

APP_IMPORT_PATH = "offline_llm_eval.main:app"
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_or_create_session_factory(app)
    try:
        yield
    finally:
        await dispose_database_engine(app)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Offline LLM Eval",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(runs_router)
    app.include_router(cases_router)
    app.include_router(diff_router)
    app.include_router(review_router)
    app.include_router(evidence_router)

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return await request_validation_error_handler(request, exc)

    return app


app = create_app()


def run_api_server() -> None:
    uvicorn.run(APP_IMPORT_PATH, host=DEFAULT_API_HOST, port=DEFAULT_API_PORT)


if __name__ == "__main__":
    run_api_server()
