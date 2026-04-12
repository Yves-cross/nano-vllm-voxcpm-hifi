from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_server
from app.schemas.http import ErrorResponse, InfoResponse, LoRAInfo, ModelInfo, Mp3Info

router = APIRouter(tags=["info"])


@router.get(
    "/info",
    response_model=InfoResponse,
    summary="Get model and service metadata",
    responses={
        503: {
            "description": "Model server not ready",
            "model": ErrorResponse,
        }
    },
)
async def info(request: Request, server: Any = Depends(get_server)) -> InfoResponse:
    """Return model metadata and instance-level configuration."""

    cfg = getattr(request.app.state, "cfg", None)
    model_info = await server.get_model_info()

    lora_state = getattr(request.app.state, "lora", {})
    return InfoResponse(
        model=ModelInfo(
            sample_rate=int(model_info["sample_rate"]),
            channels=int(model_info["channels"]),
            feat_dim=int(model_info["feat_dim"]),
            patch_size=int(model_info["patch_size"]),
            model_path=str(model_info["model_path"]),
        ),
        lora=LoRAInfo(
            lora_uri=lora_state.get("lora_uri"),
            lora_id=lora_state.get("lora_id"),
            cache_dir=lora_state.get("cache_dir"),
            loaded=bool(lora_state.get("loaded", False)),
        ),
        mp3=Mp3Info(
            bitrate_kbps=getattr(getattr(cfg, "mp3", None), "bitrate_kbps", None),
            quality=getattr(getattr(cfg, "mp3", None), "quality", None),
        ),
    )
