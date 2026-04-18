from __future__ import annotations

import base64
import logging
from time import perf_counter
from typing import Any

from app.core.metrics import ADD_REFERENCE_TOTAL_SECONDS

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_server
from app.schemas.http import AddReferenceRequest, AddReferenceResponse, ErrorResponse

router = APIRouter(tags=["references"])
logger = logging.getLogger("uvicorn.error")


@router.post(
    "/add_reference",
    response_model=AddReferenceResponse,
    summary="Cache reference audio and return a reusable reference_id",
    responses={
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
        500: {"description": "Internal error", "model": ErrorResponse},
    },
)
async def add_reference(req: AddReferenceRequest, server: Any = Depends(get_server)) -> AddReferenceResponse:
    try:
        wav = base64.b64decode(req.wav_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 in wav_base64: {e}") from e

    t0 = perf_counter()
    reference_id = await server.add_reference(wav, req.wav_format)
    model_info = await server.get_model_info()
    total_sec = perf_counter() - t0
    ADD_REFERENCE_TOTAL_SECONDS.observe(total_sec)
    logger.info(
        "add_reference_trace wav_bytes=%s wav_format=%s total=%.4f reference_id=%s",
        len(wav),
        req.wav_format,
        total_sec,
        reference_id,
    )
    return AddReferenceResponse(
        reference_id=reference_id,
        feat_dim=int(model_info["feat_dim"]),
        sample_rate=int(model_info["sample_rate"]),
        channels=int(model_info["channels"]),
    )


@router.delete(
    "/references/{reference_id}",
    summary="Delete a cached reference_id",
    responses={
        204: {"description": "Deleted"},
        404: {"description": "Not found", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
    },
)
async def delete_reference(reference_id: str, server: Any = Depends(get_server)):
    try:
        await server.remove_reference(reference_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Reference with id {reference_id} not found") from e
    return None
