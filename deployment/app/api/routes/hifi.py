from __future__ import annotations

import base64
import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_server
from app.schemas.http import AddHiFiRequest, AddHiFiResponse, ErrorResponse

router = APIRouter(tags=["hifi"])
logger = logging.getLogger("uvicorn.error")


def _get_hifi_pool(request: Request) -> dict[str, dict[str, str]]:
    pool = getattr(request.app.state, "hifi_pool", None)
    if pool is None:
        pool = {}
        request.app.state.hifi_pool = pool
    return pool


@router.post(
    "/add_hifi",
    response_model=AddHiFiResponse,
    summary="Cache a HiFi clone bundle (prompt_id + reference_id) and return a reusable hifi_id",
    responses={400: {"description": "Invalid input", "model": ErrorResponse}, 503: {"description": "Model server not ready", "model": ErrorResponse}, 500: {"description": "Internal error", "model": ErrorResponse}},
)
async def add_hifi(req: AddHiFiRequest, request: Request, server: Any = Depends(get_server)) -> AddHiFiResponse:
    try:
        wav = base64.b64decode(req.wav_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 in wav_base64: {e}") from e

    t0 = perf_counter()
    prompt_id = await server.add_prompt(wav, req.wav_format, req.prompt_text)
    reference_id = await server.add_reference(wav, req.wav_format)
    hifi_id = uuid4().hex
    pool = _get_hifi_pool(request)
    pool[hifi_id] = {"prompt_id": prompt_id, "reference_id": reference_id}
    model_info = await server.get_model_info()
    total_sec = perf_counter() - t0
    logger.info(
        "add_hifi_trace wav_bytes=%s wav_format=%s prompt_text_len=%s total=%.4f hifi_id=%s prompt_id=%s reference_id=%s",
        len(wav), req.wav_format, len(req.prompt_text), total_sec, hifi_id, prompt_id, reference_id
    )
    return AddHiFiResponse(
        hifi_id=hifi_id,
        prompt_id=prompt_id,
        reference_id=reference_id,
        feat_dim=int(model_info["feat_dim"]),
        sample_rate=int(model_info["sample_rate"]),
        channels=int(model_info["channels"]),
    )


@router.delete(
    "/hifi/{hifi_id}",
    summary="Delete a cached hifi_id (and its underlying prompt/reference caches)",
    responses={204: {"description": "Deleted"}, 404: {"description": "Not found", "model": ErrorResponse}, 503: {"description": "Model server not ready", "model": ErrorResponse}},
)
async def delete_hifi(hifi_id: str, request: Request, server: Any = Depends(get_server)):
    pool = _get_hifi_pool(request)
    if hifi_id not in pool:
        raise HTTPException(status_code=404, detail=f"HiFi with id {hifi_id} not found")
    item = pool.pop(hifi_id)
    try:
        await server.remove_prompt(item["prompt_id"])
    except KeyError:
        pass
    try:
        await server.remove_reference(item["reference_id"])
    except KeyError:
        pass
    return None
