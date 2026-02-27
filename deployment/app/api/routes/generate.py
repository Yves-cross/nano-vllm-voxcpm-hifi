from __future__ import annotations

import base64
import time
from typing import AsyncIterator

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from numpy.typing import NDArray

from app.api.deps import get_server
from app.core.metrics import (
    GENERATE_AUDIO_SECONDS_TOTAL,
    GENERATE_STREAM_BYTES_TOTAL,
    GENERATE_TTFB_SECONDS,
)
from app.schemas.http import ErrorResponse, GenerateRequest
from app.services.mp3 import stream_mp3
from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool

router = APIRouter(tags=["generation"])


def _validate_generate_prompt(req: GenerateRequest) -> None:
    has_wav = req.prompt_wav_base64 is not None or req.prompt_wav_format is not None
    has_latents = req.prompt_latents_base64 is not None

    if has_wav and has_latents:
        raise HTTPException(
            status_code=400,
            detail="prompt_wav_* and prompt_latents_base64 are mutually exclusive",
        )

    if has_wav:
        if req.prompt_wav_base64 is None or req.prompt_wav_format is None:
            raise HTTPException(
                status_code=400,
                detail="wav prompt requires prompt_wav_base64 + prompt_wav_format",
            )
        if req.prompt_text is None or req.prompt_text == "":
            raise HTTPException(status_code=400, detail="wav prompt requires non-empty prompt_text")
        return

    if has_latents:
        if req.prompt_text is None or req.prompt_text == "":
            raise HTTPException(status_code=400, detail="latents prompt requires non-empty prompt_text")
        return

    if req.prompt_text not in (None, ""):
        raise HTTPException(status_code=400, detail="prompt_text is not allowed for zero-shot")


@router.post(
    "/generate",
    response_class=StreamingResponse,
    summary="Generate audio (streaming MP3)",
    responses={
        200: {
            "description": "MP3 byte stream",
            "content": {
                "audio/mpeg": {
                    "schema": {"type": "string", "format": "binary"},
                }
            },
            "headers": {
                "X-Audio-Sample-Rate": {
                    "description": "Audio sample rate in Hz.",
                    "schema": {"type": "integer"},
                },
                "X-Audio-Channels": {
                    "description": "Number of audio channels.",
                    "schema": {"type": "integer"},
                },
            },
        },
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
        500: {"description": "Internal error", "model": ErrorResponse},
    },
)
async def generate(
    req: GenerateRequest,
    request: Request,
    server: AsyncVoxCPMServerPool = Depends(get_server),
) -> StreamingResponse:
    """Generate speech audio as a streamed MP3 byte stream.

    The response is streamed and may terminate early if the client disconnects or
    an internal error occurs after streaming has started.
    """

    _validate_generate_prompt(req)

    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=500, detail="server misconfigured: missing app.state.cfg")

    model_info = await server.get_model_info()
    sample_rate = int(model_info["sample_rate"])
    channels = int(model_info["channels"])
    if channels != 1:
        raise HTTPException(status_code=500, detail=f"Only mono is supported (channels={channels})")

    prompt_latents: bytes | None = None
    prompt_text = ""
    if req.prompt_wav_base64 is not None:
        try:
            wav = base64.b64decode(req.prompt_wav_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in prompt_wav_base64: {e}") from e
        assert req.prompt_wav_format is not None
        assert req.prompt_text is not None
        prompt_latents = await server.encode_latents(wav, req.prompt_wav_format)
        prompt_text = req.prompt_text
    elif req.prompt_latents_base64 is not None:
        try:
            prompt_latents = base64.b64decode(req.prompt_latents_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in prompt_latents_base64: {e}") from e
        assert req.prompt_text is not None
        prompt_text = req.prompt_text

    start_t = time.perf_counter()
    ttfb_recorded = False

    async def wav_chunks() -> AsyncIterator[NDArray[np.float32]]:
        async for chunk in server.generate(
            target_text=req.target_text,
            prompt_latents=prompt_latents,
            prompt_text=prompt_text,
            max_generate_length=req.max_generate_length,
            temperature=req.temperature,
            cfg_value=req.cfg_value,
        ):
            GENERATE_AUDIO_SECONDS_TOTAL.inc(float(chunk.shape[0]) / float(sample_rate))
            yield chunk

    async def body() -> AsyncIterator[bytes]:
        nonlocal ttfb_recorded
        async for b in stream_mp3(
            request=request,
            wav_chunks=wav_chunks(),
            sample_rate=sample_rate,
            mp3=cfg.mp3,
        ):
            if not ttfb_recorded:
                GENERATE_TTFB_SECONDS.observe(time.perf_counter() - start_t)
                ttfb_recorded = True
            GENERATE_STREAM_BYTES_TOTAL.inc(len(b))
            yield b
        if not ttfb_recorded:
            GENERATE_TTFB_SECONDS.observe(time.perf_counter() - start_t)

    return StreamingResponse(
        body(),
        media_type="audio/mpeg",
        headers={
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Audio-Channels": str(channels),
        },
    )
