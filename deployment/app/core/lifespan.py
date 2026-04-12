from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from nanovllm_voxcpm.llm import VoxCPM
from nanovllm_voxcpm.models.voxcpm.config import LoRAConfig as VoxLoRAConfig

from app.core.config import ServiceConfig
from app.core.metrics import LORA_LOADED, LORA_LOAD_SECONDS
from app.services.lora_resolver import (
    load_lora_config_from_checkpoint,
    normalize_lora_checkpoint_path,
    resolve_lora_uri,
)

SERVER_FACTORY = VoxCPM.from_pretrained
logger = logging.getLogger("uvicorn.error")


def build_lifespan(cfg: ServiceConfig):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        lora_config: VoxLoRAConfig | None = None
        lora_ckpt_path: str | None = None

        if cfg.lora.uri:
            # Resolve before starting the model so we can honor LoRAConfig saved in lora_config.json.
            resolved = resolve_lora_uri(
                uri=cfg.lora.uri,
                cache_dir=cfg.lora.cache_dir,
                expected_sha256=cfg.lora.sha256,
            )
            ckpt_dir = normalize_lora_checkpoint_path(resolved.local_path)
            lora_ckpt_path = str(ckpt_dir)
            parsed_cfg = load_lora_config_from_checkpoint(ckpt_dir)
            lora_config = parsed_cfg or VoxLoRAConfig()

        server = SERVER_FACTORY(
            model=cfg.model_path,
            inference_timesteps=cfg.server_pool.inference_timesteps,
            max_num_batched_tokens=cfg.server_pool.max_num_batched_tokens,
            max_num_seqs=cfg.server_pool.max_num_seqs,
            max_model_len=cfg.server_pool.max_model_len,
            gpu_memory_utilization=cfg.server_pool.gpu_memory_utilization,
            enforce_eager=cfg.server_pool.enforce_eager,
            devices=list(cfg.server_pool.devices),
            lora_config=lora_config,
        )
        app.state.server = server
        app.state.ready = False
        app.state.lora = {
            "lora_uri": cfg.lora.uri,
            "lora_id": cfg.lora.lora_id,
            "cache_dir": cfg.lora.cache_dir,
            "loaded": False,
        }

        warmup_task: asyncio.Task | None = None

        async def _run_async_warmup() -> None:
            try:
                if cfg.warmup.delay_sec > 0:
                    await asyncio.sleep(cfg.warmup.delay_sec)
                t0 = time.perf_counter()
                async for _chunk in server.generate(
                    target_text=cfg.warmup.text,
                    max_generate_length=cfg.warmup.max_generate_length,
                ):
                    logger.info("async_warmup_trace status=first_chunk_ok delay=%.3f elapsed=%.4f", cfg.warmup.delay_sec, time.perf_counter() - t0)
                    break
                else:
                    logger.info("async_warmup_trace status=no_chunk delay=%.3f elapsed=%.4f", cfg.warmup.delay_sec, time.perf_counter() - t0)
            except asyncio.CancelledError:
                logger.info("async_warmup_trace status=cancelled")
                raise
            except Exception as e:
                logger.warning("async_warmup_trace status=failed delay=%.3f err=%s", cfg.warmup.delay_sec, e)

        try:
            await server.wait_for_ready()

            if cfg.lora.uri:
                t0 = time.perf_counter()
                assert lora_ckpt_path is not None
                await server.load_lora(lora_ckpt_path)
                await server.set_lora_enabled(True)
                LORA_LOADED.labels(lora_id=str(cfg.lora.lora_id)).set(1)
                LORA_LOAD_SECONDS.observe(time.perf_counter() - t0)
                app.state.lora.update({"loaded": True})

            app.state.ready = True
            if cfg.warmup.enabled:
                warmup_task = asyncio.create_task(_run_async_warmup(), name="nanovllm-async-warmup")
                app.state.warmup_task = warmup_task
                logger.info("async_warmup_trace status=scheduled delay=%.3f", cfg.warmup.delay_sec)
            yield
        finally:
            if warmup_task is not None and not warmup_task.done():
                warmup_task.cancel()
                try:
                    await warmup_task
                except asyncio.CancelledError:
                    pass
            app.state.ready = False
            await server.stop()
            if getattr(app.state, "server", None) is server:
                delattr(app.state, "server")

    return lifespan
