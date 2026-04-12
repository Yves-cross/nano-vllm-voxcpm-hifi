from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import uuid
import wave
from typing import Any, AsyncIterator

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from numpy.typing import NDArray

from app.api.deps import get_server
from app.core.metrics import (
    GENERATE_AUDIO_SECONDS_TOTAL,
    GENERATE_CONTEXT_RESOLVE_SECONDS,
    GENERATE_FIRST_MP3_CHUNK_SECONDS,
    GENERATE_FIRST_WAV_CHUNK_SECONDS,
    GENERATE_STREAM_BYTES_TOTAL,
    GENERATE_TOTAL_SECONDS,
    GENERATE_TTFB_SECONDS,
)
from app.schemas.http import ErrorResponse, GenerateRequest
from app.services.mp3 import encode_mp3_bytes, float32_to_s16le_bytes, stream_mp3

router = APIRouter(tags=["generation"])
logger = logging.getLogger("uvicorn.error")


def _decode_latents_base64(value: str, field_name: str, feat_dim: int) -> bytes:
    try:
        latents = base64.b64decode(value)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 in {field_name}: {e}") from e

    try:
        np.frombuffer(latents, dtype=np.float32).reshape(-1, feat_dim)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid latent payload in {field_name}: {e}") from e

    return latents


def _validate_generate_prompt(req: GenerateRequest) -> None:
    has_wav = req.prompt_wav_base64 is not None or req.prompt_wav_format is not None
    has_latents = req.prompt_latents_base64 is not None
    has_prompt_id = req.prompt_id is not None
    has_hifi_id = req.hifi_id is not None
    has_ref_wav = req.ref_audio_wav_base64 is not None or req.ref_audio_wav_format is not None
    has_ref_latents = req.ref_audio_latents_base64 is not None
    has_ref_id = req.ref_audio_id is not None

    if sum([1 if has_wav else 0, 1 if has_latents else 0, 1 if has_prompt_id else 0, 1 if has_hifi_id else 0]) > 1:
        raise HTTPException(
            status_code=400,
            detail="prompt forms (wav/latents/id) are mutually exclusive",
        )

    if sum([1 if has_ref_wav else 0, 1 if has_ref_latents else 0, 1 if has_ref_id else 0]) > 1:
        raise HTTPException(
            status_code=400,
            detail="reference audio forms (wav/latents/id) are mutually exclusive",
        )

    if has_ref_wav and (req.ref_audio_wav_base64 is None or req.ref_audio_wav_format is None):
        raise HTTPException(
            status_code=400,
            detail="reference wav requires ref_audio_wav_base64 + ref_audio_wav_format",
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

    if has_prompt_id:
        if req.prompt_text not in (None, ""):
            raise HTTPException(status_code=400, detail="prompt_text is not allowed when prompt_id is set")
        return

    if has_hifi_id:
        if req.prompt_text not in (None, ""):
            raise HTTPException(status_code=400, detail="prompt_text is not allowed when hifi_id is set")
        return

    if req.prompt_text not in (None, ""):
        raise HTTPException(status_code=400, detail="prompt_text is not allowed for zero-shot")


async def _resolve_generation_context(req: GenerateRequest, server: Any, feat_dim: int, hifi_pool: dict[str, dict[str, str]] | None = None) -> tuple[bytes | None, bytes | None, str, str | None, str | None]:
    prompt_latents: bytes | None = None
    ref_audio_latents: bytes | None = None
    prompt_text = ""
    prompt_id: str | None = None
    ref_audio_id: str | None = None

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
        prompt_latents = _decode_latents_base64(req.prompt_latents_base64, "prompt_latents_base64", feat_dim)
        assert req.prompt_text is not None
        prompt_text = req.prompt_text
    elif req.prompt_id is not None:
        prompt_id = req.prompt_id
    elif req.hifi_id is not None:
        if hifi_pool is None or req.hifi_id not in hifi_pool:
            raise HTTPException(status_code=404, detail=f"HiFi with id {req.hifi_id} not found")
        item = hifi_pool[req.hifi_id]
        prompt_id = item["prompt_id"]
        ref_audio_id = item["reference_id"]

    if req.ref_audio_wav_base64 is not None:
        try:
            wav = base64.b64decode(req.ref_audio_wav_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in ref_audio_wav_base64: {e}") from e
        assert req.ref_audio_wav_format is not None
        ref_audio_latents = await server.encode_latents(wav, req.ref_audio_wav_format)
    elif req.ref_audio_latents_base64 is not None:
        ref_audio_latents = _decode_latents_base64(req.ref_audio_latents_base64, "ref_audio_latents_base64", feat_dim)
    elif req.ref_audio_id is not None:
        ref_audio_id = req.ref_audio_id

    return prompt_latents, ref_audio_latents, prompt_text, prompt_id, ref_audio_id


def _build_generate_kwargs(req: GenerateRequest, prompt_latents: bytes | None, prompt_text: str, prompt_id: str | None, ref_audio_latents: bytes | None, ref_audio_id: str | None) -> dict[str, Any]:
    generate_kwargs = {
        "target_text": req.target_text,
        "prompt_latents": prompt_latents,
        "prompt_text": prompt_text,
        "max_generate_length": req.max_generate_length,
        "temperature": req.temperature,
        "cfg_value": req.cfg_value,
    }
    if prompt_id is not None:
        generate_kwargs["prompt_id"] = prompt_id
    if ref_audio_latents is not None:
        generate_kwargs["ref_audio_latents"] = ref_audio_latents
    if ref_audio_id is not None:
        generate_kwargs["ref_audio_id"] = ref_audio_id
    return generate_kwargs


async def _make_wav_chunks(req: GenerateRequest, server: Any, prompt_latents: bytes | None, prompt_text: str, prompt_id: str | None, ref_audio_latents: bytes | None, ref_audio_id: str | None, sample_rate: int) -> AsyncIterator[NDArray[np.float32]]:
    generate_kwargs = _build_generate_kwargs(req, prompt_latents, prompt_text, prompt_id, ref_audio_latents, ref_audio_id)
    try:
        stream = server.generate(**generate_kwargs)
    except TypeError as e:
        if ref_audio_latents is None:
            raise
        raise HTTPException(
            status_code=400, detail=f"Reference audio is not supported by the loaded model: {e}"
        ) from e

    async for chunk in stream:
        GENERATE_AUDIO_SECONDS_TOTAL.inc(float(chunk.shape[0]) / float(sample_rate))
        yield chunk


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
    server: Any = Depends(get_server),
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
    feat_dim = int(model_info["feat_dim"])
    if channels != 1:
        raise HTTPException(status_code=500, detail=f"Only mono is supported (channels={channels})")

    req_id = uuid.uuid4().hex[:8]
    route_name = "/generate"
    start_t = time.perf_counter()
    prompt_latents, ref_audio_latents, prompt_text, prompt_id, ref_audio_id = await _resolve_generation_context(req, server, feat_dim, getattr(request.app.state, "hifi_pool", None))
    resolve_sec = time.perf_counter() - start_t
    GENERATE_CONTEXT_RESOLVE_SECONDS.labels(route=route_name).observe(resolve_sec)

    ttfb_recorded = False
    first_wav_recorded = False
    first_wav_sec = None
    first_mp3_sec = None

    async def instrumented_wav_chunks() -> AsyncIterator[NDArray[np.float32]]:
        nonlocal first_wav_recorded, first_wav_sec
        async for chunk in _make_wav_chunks(req, server, prompt_latents, prompt_text, prompt_id, ref_audio_latents, ref_audio_id, sample_rate):
            if not first_wav_recorded:
                first_wav_recorded = True
                first_wav_sec = time.perf_counter() - start_t
                GENERATE_FIRST_WAV_CHUNK_SECONDS.labels(route=route_name).observe(first_wav_sec)
            yield chunk

    async def body() -> AsyncIterator[bytes]:
        nonlocal ttfb_recorded
        total_bytes = 0
        async for b in stream_mp3(
            request=request,
            wav_chunks=instrumented_wav_chunks(),
            sample_rate=sample_rate,
            mp3=cfg.mp3,
        ):
            if not ttfb_recorded:
                nonlocal first_mp3_sec
                ttfb = time.perf_counter() - start_t
                first_mp3_sec = ttfb
                GENERATE_TTFB_SECONDS.observe(ttfb)
                GENERATE_FIRST_MP3_CHUNK_SECONDS.labels(route=route_name).observe(ttfb)
                ttfb_recorded = True
            GENERATE_STREAM_BYTES_TOTAL.inc(len(b))
            total_bytes += len(b)
            yield b
        total_sec = time.perf_counter() - start_t
        if not ttfb_recorded:
            GENERATE_TTFB_SECONDS.observe(total_sec)
        GENERATE_TOTAL_SECONDS.labels(route=route_name).observe(total_sec)
        logger.info(
            "generate_trace route=%s req_id=%s text_len=%s resolve=%.4f first_wav=%s first_mp3=%s total=%.4f bytes=%s ref_mode=%s",
            route_name,
            req_id,
            len(req.target_text),
            resolve_sec,
            f"{first_wav_sec:.4f}" if first_wav_sec is not None else "na",
            f"{first_mp3_sec:.4f}" if first_mp3_sec is not None else "na",
            total_sec,
            total_bytes,
            "hifi_id" if req.hifi_id is not None else ("prompt_id" if req.prompt_id is not None else ("prompt_latents" if req.prompt_latents_base64 is not None else ("prompt_wav" if req.prompt_wav_base64 is not None else ("ref_id" if req.ref_audio_id is not None else ("ref_latents" if req.ref_audio_latents_base64 is not None else ("ref_wav" if req.ref_audio_wav_base64 is not None else "none")))) )),
        )

    return StreamingResponse(
        body(),
        media_type="audio/mpeg",
        headers={
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Audio-Channels": str(channels),
        },
    )


@router.post(
    "/generate_blocking",
    response_class=Response,
    summary="Generate audio (non-streaming MP3)",
    responses={
        200: {
            "description": "Complete MP3 bytes returned after full generation finishes",
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
                "X-Generate-Mode": {
                    "description": "Return mode for this endpoint.",
                    "schema": {"type": "string"},
                },
            },
        },
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
        500: {"description": "Internal error", "model": ErrorResponse},
    },
)
async def generate_blocking(
    req: GenerateRequest,
    request: Request,
    server: Any = Depends(get_server),
) -> Response:
    """Generate speech audio and return the complete MP3 after synthesis finishes."""

    _validate_generate_prompt(req)

    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=500, detail="server misconfigured: missing app.state.cfg")

    model_info = await server.get_model_info()
    sample_rate = int(model_info["sample_rate"])
    channels = int(model_info["channels"])
    feat_dim = int(model_info["feat_dim"])
    if channels != 1:
        raise HTTPException(status_code=500, detail=f"Only mono is supported (channels={channels})")

    req_id = uuid.uuid4().hex[:8]
    route_name = "/generate_blocking"
    start_t = time.perf_counter()
    prompt_latents, ref_audio_latents, prompt_text, prompt_id, ref_audio_id = await _resolve_generation_context(req, server, feat_dim, getattr(request.app.state, "hifi_pool", None))
    resolve_sec = time.perf_counter() - start_t
    GENERATE_CONTEXT_RESOLVE_SECONDS.labels(route=route_name).observe(resolve_sec)

    wav_parts: list[NDArray[np.float32]] = []
    first_wav_sec = None
    async for chunk in _make_wav_chunks(req, server, prompt_latents, prompt_text, prompt_id, ref_audio_latents, ref_audio_id, sample_rate):
        if first_wav_sec is None:
            first_wav_sec = time.perf_counter() - start_t
            GENERATE_FIRST_WAV_CHUNK_SECONDS.labels(route=route_name).observe(first_wav_sec)
        wav_parts.append(chunk)

    if wav_parts:
        wav = np.concatenate(wav_parts, axis=0)
    else:
        wav = np.zeros((0,), dtype=np.float32)

    encode_t0 = time.perf_counter()
    mp3_bytes = await asyncio.to_thread(encode_mp3_bytes, wav, sample_rate, cfg.mp3)
    encode_sec = time.perf_counter() - encode_t0
    total_sec = time.perf_counter() - start_t
    GENERATE_TTFB_SECONDS.observe(total_sec)
    GENERATE_FIRST_MP3_CHUNK_SECONDS.labels(route=route_name).observe(total_sec)
    GENERATE_TOTAL_SECONDS.labels(route=route_name).observe(total_sec)
    GENERATE_STREAM_BYTES_TOTAL.inc(len(mp3_bytes))
    logger.info(
        "generate_trace route=%s req_id=%s text_len=%s resolve=%.4f first_wav=%s first_mp3=%s mp3_encode=%.4f total=%.4f bytes=%s ref_mode=%s",
        route_name, req_id, len(req.target_text), resolve_sec,
        f"{first_wav_sec:.4f}" if first_wav_sec is not None else "na",
        f"{total_sec:.4f}", encode_sec, total_sec, len(mp3_bytes),
        "hifi_id" if req.hifi_id is not None else ("prompt_id" if req.prompt_id is not None else ("prompt_latents" if req.prompt_latents_base64 is not None else ("prompt_wav" if req.prompt_wav_base64 is not None else ("ref_id" if req.ref_audio_id is not None else ("ref_latents" if req.ref_audio_latents_base64 is not None else ("ref_wav" if req.ref_audio_wav_base64 is not None else "none")))) )),
    )

    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Audio-Channels": str(channels),
            "X-Generate-Mode": "blocking",
        },
    )


@router.post(
    "/generate_blocking_wav",
    response_class=Response,
    summary="Generate audio (non-streaming WAV)",
    responses={
        200: {
            "description": "Complete WAV bytes returned after full generation finishes",
            "content": {
                "audio/wav": {
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
                "X-Generate-Mode": {
                    "description": "Return mode for this endpoint.",
                    "schema": {"type": "string"},
                },
            },
        },
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
        500: {"description": "Internal error", "model": ErrorResponse},
    },
)
async def generate_blocking_wav(
    req: GenerateRequest,
    request: Request,
    server: Any = Depends(get_server),
) -> Response:
    """Generate speech audio and return the complete WAV after synthesis finishes."""

    _validate_generate_prompt(req)

    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=500, detail="server misconfigured: missing app.state.cfg")

    model_info = await server.get_model_info()
    sample_rate = int(model_info["sample_rate"])
    channels = int(model_info["channels"])
    feat_dim = int(model_info["feat_dim"])
    if channels != 1:
        raise HTTPException(status_code=500, detail=f"Only mono is supported (channels={channels})")

    req_id = uuid.uuid4().hex[:8]
    route_name = "/generate_blocking_wav"
    start_t = time.perf_counter()
    prompt_latents, ref_audio_latents, prompt_text, prompt_id, ref_audio_id = await _resolve_generation_context(req, server, feat_dim, getattr(request.app.state, "hifi_pool", None))
    resolve_sec = time.perf_counter() - start_t
    GENERATE_CONTEXT_RESOLVE_SECONDS.labels(route=route_name).observe(resolve_sec)

    wav_parts: list[NDArray[np.float32]] = []
    first_wav_sec = None
    async for chunk in _make_wav_chunks(req, server, prompt_latents, prompt_text, prompt_id, ref_audio_latents, ref_audio_id, sample_rate):
        if first_wav_sec is None:
            first_wav_sec = time.perf_counter() - start_t
            GENERATE_FIRST_WAV_CHUNK_SECONDS.labels(route=route_name).observe(first_wav_sec)
        wav_parts.append(chunk)

    if wav_parts:
        wav = np.concatenate(wav_parts, axis=0)
    else:
        wav = np.zeros((0,), dtype=np.float32)

    pcm_bytes = float32_to_s16le_bytes(wav)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    wav_bytes = bio.getvalue()

    total_sec = time.perf_counter() - start_t
    GENERATE_TOTAL_SECONDS.labels(route=route_name).observe(total_sec)
    logger.info(
        "generate_trace route=%s req_id=%s text_len=%s resolve=%.4f first_wav=%s total=%.4f bytes=%s ref_mode=%s",
        route_name, req_id, len(req.target_text), resolve_sec,
        f"{first_wav_sec:.4f}" if first_wav_sec is not None else "na",
        total_sec, len(wav_bytes),
        "hifi_id" if req.hifi_id is not None else ("prompt_id" if req.prompt_id is not None else ("prompt_latents" if req.prompt_latents_base64 is not None else ("prompt_wav" if req.prompt_wav_base64 is not None else ("ref_id" if req.ref_audio_id is not None else ("ref_latents" if req.ref_audio_latents_base64 is not None else ("ref_wav" if req.ref_audio_wav_base64 is not None else "none")))) )),
    )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Audio-Channels": str(channels),
            "X-Generate-Mode": "blocking-wav",
        },
    )
