from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from app.core.metrics import metrics_response

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    response_class=Response,
    summary="Prometheus metrics",
    responses={
        200: {
            "description": "Prometheus text exposition format",
            "content": {
                "text/plain": {
                    "schema": {
                        "type": "string",
                        "description": "Prometheus metrics in text format",
                    }
                }
            },
        }
    },
)
async def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""

    return metrics_response()
