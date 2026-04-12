from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.http import ErrorResponse, HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
)
async def health() -> HealthResponse:
    """Liveness probe."""

    return HealthResponse()


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    responses={
        503: {
            "description": "Model is still loading",
            "model": ErrorResponse,
        }
    },
)
async def ready(request: Request) -> HealthResponse:
    """Return 200 only after the model server is ready."""

    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    return HealthResponse()
