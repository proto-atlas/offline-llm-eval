from typing import TypedDict

from fastapi import APIRouter


class HealthResponse(TypedDict):
    status: str


router = APIRouter()


@router.get("/health")
def get_liveness() -> HealthResponse:
    return {"status": "ok"}


@router.get("/api/health")
def get_readiness() -> HealthResponse:
    return {"status": "ready"}
