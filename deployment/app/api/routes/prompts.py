from __future__ import annotations

import base64
import logging
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_server
from app.schemas.http import AddPromptRequest, AddPromptResponse, ErrorResponse

router = APIRouter(tags=["prompts"])
logger = logging.getLogger("uvicorn.error")


@router.post(
    "/add_prompt",
    response_model=AddPromptResponse,
    summary="Cache prompt audio+text and return a reusable prompt_id",
    responses={400: {"description": "Invalid input", "model": ErrorResponse}, 503: {"description": "Model server not ready", "model": ErrorResponse}, 500: {"description": "Internal error", "model": ErrorResponse}},
)
async def add_prompt(req: AddPromptRequest, server: Any = Depends(get_server)) -> AddPromptResponse:
    try:
        wav = base64.b64decode(req.wav_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 in wav_base64: {e}") from e

    t0 = perf_counter()
    prompt_id = await server.add_prompt(wav, req.wav_format, req.prompt_text)
    model_info = await server.get_model_info()
    total_sec = perf_counter() - t0
    logger.info(
        "add_prompt_trace wav_bytes=%s wav_format=%s prompt_text_len=%s total=%.4f prompt_id=%s",
        len(wav), req.wav_format, len(req.prompt_text), total_sec, prompt_id
    )
    return AddPromptResponse(
        prompt_id=prompt_id,
        feat_dim=int(model_info["feat_dim"]),
        sample_rate=int(model_info["sample_rate"]),
        channels=int(model_info["channels"]),
    )


@router.delete(
    "/prompts/{prompt_id}",
    summary="Delete a cached prompt_id",
    responses={204: {"description": "Deleted"}, 404: {"description": "Not found", "model": ErrorResponse}, 503: {"description": "Model server not ready", "model": ErrorResponse}},
)
async def delete_prompt(prompt_id: str, server: Any = Depends(get_server)):
    try:
        await server.remove_prompt(prompt_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Prompt with id {prompt_id} not found") from e
    return None
