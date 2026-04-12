from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.encode_latents import router as encode_latents_router
from app.api.routes.generate import router as generate_router
from app.api.routes.references import router as references_router
from app.api.routes.health import router as health_router
from app.api.routes.hifi import router as hifi_router
from app.api.routes.info import router as info_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.prompts import router as prompts_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(info_router)
api_router.include_router(metrics_router)
api_router.include_router(encode_latents_router)
api_router.include_router(generate_router)
api_router.include_router(prompts_router)
api_router.include_router(hifi_router)
api_router.include_router(references_router)
