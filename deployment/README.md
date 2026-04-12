# VoxCPM FastAPI Service

This folder contains a production-oriented FastAPI wrapper around
`nanovllm_voxcpm.models.voxcpm.server.AsyncVoxCPMServerPool`.

Key properties:

- Stateful cache endpoints for `prompt_id`, `reference_id`, and `hifi_id`
- No runtime LoRA management endpoints
- `/generate` streams MP3 (`audio/mpeg`) encoded server-side via `lameenc`
- `/generate_blocking` and `/generate_blocking_wav` provide non-streaming responses

## Install (uv)

This repo uses `uv` and `deployment/` is a uv workspace member.

Install workspace dependencies at the repo root:

```bash
uv sync --all-packages --frozen
```

Alternatively, to sync only the deployment service dependencies:

```bash
uv sync --package nano-vllm-voxcpm-deployment --frozen
```

Note: `uv sync --frozen` (without `--all-packages/--package`) only syncs the root package by default.

## Configure

Environment variables:

- `NANOVLLM_MODEL_PATH` (default `~/VoxCPM1.5`)
- MP3 encoding (read at startup):
  - `NANOVLLM_MP3_BITRATE_KBPS` (int, default `192`)
  - `NANOVLLM_MP3_QUALITY` (int, default `2`, allowed `0..2`)
- LoRA startup (optional; instance-level fixed, no runtime switching):
  - `NANOVLLM_LORA_URI` (examples: `file:///...`, `https://...`, `s3://bucket/key`, `hf://repo@rev?path=...`)
  - `NANOVLLM_LORA_ID` (required if `NANOVLLM_LORA_URI` is set)
  - `NANOVLLM_LORA_SHA256` (optional; full-file checksum)
  - `NANOVLLM_CACHE_DIR` (default `~/.cache/nanovllm`)

- Server pool startup (read at startup):
- Queue / warmup / startup behavior (read at startup):
  - `NANOVLLM_SERVERPOOL_INFERENCE_TIMESTEPS` (int, default `10`)
  - `NANOVLLM_QUEUE_COALESCE_MS` (int, default host-tuned; current recommended value for HiFi on RTX 4090 is `5`)
  - `NANOVLLM_WARMUP_MODE` (`zero` or `hifi`)
  - `NANOVLLM_WARMUP_TEXT`
  - `NANOVLLM_WARMUP_MAX_GENERATE_LENGTH`
  - `NANOVLLM_HIFI_WARMUP_WAV_PATH`
  - `NANOVLLM_HIFI_WARMUP_PROMPT_TEXT`
  - `NANOVLLM_HIFI_WARMUP_TARGET_TEXT`
  - `NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS` (int, default `8192`)
  - `NANOVLLM_SERVERPOOL_MAX_NUM_SEQS` (int, default `16`)
  - `NANOVLLM_SERVERPOOL_MAX_MODEL_LEN` (int, default `4096`)
  - `NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION` (float, default `0.95`, allowed `(0, 1]`)
  - `NANOVLLM_SERVERPOOL_ENFORCE_EAGER` (bool, default `false`; accepts `1/0,true/false,yes/no,on/off`)
  - `NANOVLLM_SERVERPOOL_DEVICES` (comma-separated ints, default `0`; e.g. `0,1`)

LoRA checkpoint layout (recommended):

```
step_0002000/
  lora_weights.safetensors
  lora_config.json
```

If `lora_config.json` exists, the service will read `lora_config` from it to initialize LoRA structure.

## Run

From the repo root:

```bash
uv run fastapi run deployment/app/main.py --host 0.0.0.0 --port 8000
```

Alternatively (matches the container entrypoint):

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

OpenAPI:

- http://localhost:8000/docs

## Tests

```bash
uv run pytest deployment/tests -q
```

## Docker (k8s-ready)

This repo ships a multi-stage CUDA image at `deployment/Dockerfile`.

Build from the repo root (important: build context is `.`):

```bash
docker build -f deployment/Dockerfile -t nano-vllm-voxcpm-deployment:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -e NANOVLLM_MODEL_PATH=/models/VoxCPM1.5 \
  -e NANOVLLM_CACHE_DIR=/var/cache/nanovllm \
  -v /path/to/models:/models \
  nano-vllm-voxcpm-deployment:latest
```

Notes:

- GPU: on a GPU node you typically need `--gpus all` (Docker) or the NVIDIA device plugin (k8s).
- The container runs as a non-root user (uid `10001`) and uses `NANOVLLM_CACHE_DIR` for writable cache.
- Probes: use `GET /health` (liveness) and `GET /ready` (readiness).

## Client example

`deployment/client.py` demonstrates calling `/encode_latents` and `/generate` and writes MP3 files:

It expects a prompt audio file at `deployment/prompt_audio.wav`.

```bash
uv run python deployment/client.py
```

Outputs:

- `out_zero_shot.mp3`
- `out_prompted.mp3`

## API

### Health

- `GET /health` (liveness): returns `{"status":"ok"}`
- `GET /ready` (readiness): returns 200 only after the model is loaded

### Info

`GET /info`

Returns model metadata from core (`sample_rate/channels/feat_dim/...`) plus MP3 encoder config.

### Metrics

`GET /metrics`

Prometheus metrics, including route-level TTFB / total / MP3 encode histograms added during the RTX 4090 tuning work.

### Encode prompt wav to latents

`POST /encode_latents`

Request body (JSON):

- `wav_base64`: base64-encoded bytes of the *entire audio file* (not a data URI)
- `wav_format`: container format for decoding (e.g. `wav`, `flac`, `mp3`; passed to torchaudio)

Response body (JSON):

- `prompt_latents_base64`: base64-encoded float32 bytes
- `feat_dim`: reshape with `np.frombuffer(bytes, np.float32).reshape(-1, feat_dim)`
- `latents_dtype`: `"float32"`
- `sample_rate`: output sample rate (from the model)
- `channels`: `1`

### Cache prompt / reference / hifi bundles

- `POST /add_prompt` → returns `prompt_id`
- `DELETE /prompts/{prompt_id}`
- `POST /add_reference` → returns `reference_id`
- `DELETE /references/{reference_id}`
- `POST /add_hifi` → returns `hifi_id`, `prompt_id`, and `reference_id`
- `DELETE /hifi/{hifi_id}`

`hifi_id` is the web-aligned “Ultimate Cloning / 极致克隆” cache handle. Internally it bundles one prompt cache plus one reference-audio cache built from the same uploaded audio and prompt text.

### Generate (streaming MP3)

`POST /generate`

Request body (JSON):

- `target_text`: required
- Prompt (optional, mutually exclusive):
  - wav prompt: `prompt_wav_base64` + `prompt_wav_format` + `prompt_text`
  - latents prompt: `prompt_latents_base64` + `prompt_text`
  - cached prompt: `prompt_id`
  - cached HiFi bundle: `hifi_id`
  - zero-shot: omit all prompt fields
- Reference audio (optional, mutually exclusive):
  - wav reference: `ref_audio_wav_base64` + `ref_audio_wav_format`
  - latents reference: `ref_audio_latents_base64`
  - cached reference: `ref_audio_id`
- Generation args:
  - `max_generate_length`
  - `temperature`
  - `cfg_value`

For web-aligned HiFi cloning, use the same uploaded wav for both prompt and reference roles, plus `prompt_text`, or pre-cache that bundle with `hifi_id`.

Response:

- `Content-Type: audio/mpeg`
- body is a streamed MP3 byte stream
- headers:
  - `X-Audio-Sample-Rate`
  - `X-Audio-Channels`

### Generate (non-streaming)

- `POST /generate_blocking` → returns complete `audio/mpeg`
- `POST /generate_blocking_wav` → returns complete `audio/wav`

Both endpoints accept the same request body shape as `/generate`, including `hifi_id`.

### Frontend helper

This repo now also includes a Gradio frontend example for the FastAPI HiFi route:

- `tools/nanovllm_hifi_gradio.py`
- `tools/nanovllm-hifi-gradio.service.example`

This page is intended to mimic the original VoxCPM2 “Ultimate Cloning / 极致克隆” workflow while calling Nano-vLLM FastAPI under the hood.
