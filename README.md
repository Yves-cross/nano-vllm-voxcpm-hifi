# Nano-vLLM-VoxCPM

An inference engine for VoxCPM based on Nano-vLLM.

Features:
- Faster than the pytorch implementation
- Support concurrent requests
- Friendly async API (can be wrapped by an HTTP server; see `deployment/README.md`)

This repository contains a Python package (`nanovllm_voxcpm/`) plus an optional FastAPI demo.
## What is added in this branch

This branch is no longer just a minimal async wrapper. It now includes a production-oriented FastAPI layer and a set of deployment / benchmarking helpers for real VoxCPM2 cloning use cases.

### New HTTP capabilities

The FastAPI layer now supports:

- streaming generation
  - `POST /generate`
- non-streaming generation
  - `POST /generate_blocking` (returns `audio/mpeg`)
  - `POST /generate_blocking_wav` (returns `audio/wav`)
- prompt caching
  - `POST /add_prompt`
  - `DELETE /prompts/{prompt_id}`
- reference-audio caching
  - `POST /add_reference`
  - `DELETE /references/{reference_id}`
- HiFi clone bundle caching
  - `POST /add_hifi`
  - `DELETE /hifi/{hifi_id}`

`GenerateRequest` now supports all of the following conditioning paths:

- zero-shot
- `prompt_wav_* + prompt_text`
- `prompt_latents_base64 + prompt_text`
- `prompt_id`
- `ref_audio_wav_*`
- `ref_audio_latents_base64`
- `ref_audio_id`
- `hifi_id`

### Web-aligned HiFi clone semantics

The original VoxCPM2 Gradio UI's “Ultimate Cloning / 极致克隆” is **not** equivalent to a plain `prompt_id` test, and it is also **not** equivalent to a plain `ref_audio_id` test.

The web UI effectively combines the same reference audio in two roles:

- `reference_wav_path` → separate reference-audio condition
- `prompt_wav_path + prompt_text` → continuation-style conditioning

The FastAPI equivalent is therefore:

```json
{
  "target_text": "...",
  "prompt_wav_base64": "...",
  "prompt_wav_format": "wav",
  "prompt_text": "...",
  "ref_audio_wav_base64": "...",
  "ref_audio_wav_format": "wav"
}
```

To make this cheaper and reusable, this branch adds `hifi_id`, which internally binds one `prompt_id` plus one `reference_id` into a single reusable cache handle.

### Included helper scripts

- `tools/bench_latent_concurrency.py`
- `tools/bench_prompt_vs_ref_hot.py`
- `tools/bench_prompt_family_hot.py`
- `tools/bench_hifi_concurrency.py`
- `tools/bench_hifi_blocking_concurrency.py`
- `tools/post_start_warmup.sh`
- `tools/nanovllm_hifi_gradio.py`
- `tools/nanovllm-hifi-gradio.service.example`

### Deployment / test notes from the RTX 4090 host (2026-04-12)

Host used during validation:

- OS: Ubuntu 24.04
- GPU: NVIDIA GeForce RTX 4090
- Driver: `590.48.01`
- CUDA reported by `nvidia-smi`: `13.1`

#### Main deployment pitfalls encountered

1. **`uv sync --frozen` + build isolation was not enough for `flash-attn` on this host**  
   The reliable path here was installing the repo editable without build isolation and then compiling/installing `flash-attn` explicitly.

2. **Original VoxCPM2 / Gradio side had a torch / torchvision mismatch**  
   This was fixed by moving `torchvision` to a version matching the installed CUDA/torch line.

3. **The original VoxCPM2 environment needed a newer NVIDIA driver**  
   `nvidia-driver-590` was required to stabilize the CUDA 13 line used by that side.

4. **Warmup inside FastAPI lifespan was unsafe**  
   Doing generation directly inside startup triggered `scheduler.py: assert scheduled_seqs`. The safe solution was a post-start warmup script executed after `/health` became ready.

5. **`prompt_id` and `ref_audio_id` should not be compared as if they were the same task**  
   `prompt_id` is continuation-style conditioning. `ref_audio_id` is a separate reference-audio condition. The real web “HiFi / 极致克隆” path is the combined route described above.

#### Latest stable configuration used for HiFi testing

- `NANOVLLM_SERVERPOOL_ENFORCE_EAGER=false`
- `NANOVLLM_SERVERPOOL_INFERENCE_TIMESTEPS=10`
- `NANOVLLM_QUEUE_COALESCE_MS=5`
- FastAPI default `cfg_value=2.0`
- HiFi post-start warmup enabled via `tools/post_start_warmup.sh`

#### Latest measured HiFi numbers (strict warm procedure)

Warm procedure used before concurrency tests:

1. service warmup
2. `add_hifi`
3. sleep 3s
4. warm the same `hifi_id` twice
5. sleep 5s
6. run 5-concurrency test
7. sleep 5s
8. run 10-concurrency test

With `NANOVLLM_QUEUE_COALESCE_MS=5`:

| scenario | avg TTFB (s) | avg total (s) | P95 total (s) |
|---|---:|---:|---:|
| HiFi warm single request | 0.185~0.220 | 0.696~0.722 | - |
| HiFi streaming, 5 concurrency | 0.246 | 1.078 | 1.158 |
| HiFi streaming, 10 concurrency | 0.401 | 1.582 | 1.682 |
| HiFi blocking MP3, 5 concurrency | - | 0.998 | 1.091 |
| HiFi blocking MP3, 10 concurrency | - | 1.296 | 1.426 |

#### Queue coalescing comparison under the same strict HiFi method

| `NANOVLLM_QUEUE_COALESCE_MS` | 5-concurrency avg total (s) | 10-concurrency avg total (s) | verdict |
|---:|---:|---:|---|
| 2 | 1.430 | 2.013 | slower than 5 |
| 5 | 1.078 | 1.582 | best overall |
| 10 | 1.131 | 2.421 | helps 5-conc a bit, hurts 10-conc badly |

Current recommendation for HiFi on this host: **`NANOVLLM_QUEUE_COALESCE_MS=5`**.

## Installation

### Install from PyPI

Core package:

```bash
pip install nano-vllm-voxcpm
```

Or with `uv`:

```bash
uv pip install nano-vllm-voxcpm
```

Note: the optional FastAPI demo service (`deployment/`) is not published on PyPI.

### Prerequisites

- Linux + NVIDIA GPU (CUDA)
- Python >= 3.10
- `flash-attn` is required (the package imports it at runtime)

The runtime is GPU-centric (Triton + FlashAttention). CPU-only execution is not supported.

### Install from source (dev)

This repo uses `uv` and includes a lockfile (`uv.lock`).

```bash
uv sync --frozen
```

Dev deps (tests):

```bash
uv sync --frozen --dev
```

Note: `flash-attn` may require additional system CUDA tooling depending on your environment.

## Basic Usage

See `example.py` for an end-to-end async example.

Quickstart:

```bash
uv run python example.py
```

### Load a model

`VoxCPM.from_pretrained(...)` accepts either:

- a local model directory path, or
- a HuggingFace repo id (it will download via `huggingface_hub.snapshot_download`).

The model directory is expected to contain:

- `config.json`
- one or more `*.safetensors` weight files
- `audiovae.pth` (VAE weights)

### Generate (async)

If you call `from_pretrained()` inside an async event loop, it returns an `AsyncVoxCPMServerPool`.

```python
import asyncio
import numpy as np

from nanovllm_voxcpm import VoxCPM


async def main() -> None:
    server = VoxCPM.from_pretrained(
        model="/path/to/VoxCPM",
        devices=[0],
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        gpu_memory_utilization=0.95,
    )
    await server.wait_for_ready()

    chunks = []
    async for chunk in server.generate(target_text="Hello world"):
        chunks.append(chunk)  # each chunk is a float32 numpy array

    wav = np.concatenate(chunks, axis=0)
    # Write with the model's sample rate (see your model's AudioVAE config; often 16000)
    # import soundfile as sf; sf.write("out.wav", wav, sample_rate)

    await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

### Generate (sync)

If you call `from_pretrained()` outside an event loop, it returns a `SyncVoxCPMServerPool`.

```python
import numpy as np

from nanovllm_voxcpm import VoxCPM


server = VoxCPM.from_pretrained(model="/path/to/VoxCPM", devices=[0])
chunks = []
for chunk in server.generate(target_text="Hello world"):
    chunks.append(chunk)
wav = np.concatenate(chunks, axis=0)
server.stop()
```

### Prompting and reference audio (optional)

The VoxCPM2 server supports these conditioning inputs:

- zero-shot: no prompt or reference audio
- prompt continuation: provide `prompt_latents` + `prompt_text`
- stored prompt: provide a `prompt_id` (via `add_prompt`) and then generate with that id
- reference audio: provide `ref_audio_latents` to add a separate reference-audio condition

`ref_audio_latents` is independent from `prompt_latents`:

- use `prompt_latents` when you want to continue from an existing audio prefix
- use `ref_audio_latents` when you want to provide extra reference audio without treating it as the decode prefix
            
See the public API in `nanovllm_voxcpm/models/voxcpm2/server.py` for details.




## FastAPI demo

The HTTP server demo is documented separately to keep this README focused:

- `deployment/README.md`

If you want the deployment server dependencies too, use:

```bash
uv sync --all-packages --frozen
```

## Benchmark

The `benchmark/` directory contains an end-to-end inference benchmark that drives
the public server API and reports throughput/latency metrics.

Quick run:

```bash
uv run python benchmark/bench_inference.py --model ~/VoxCPM1.5 --devices 0 --concurrency 1 --warmup 1 --iters 5
```

Use a longer English prompt (~100 words) for more stable results:

```bash
uv run python benchmark/bench_inference.py --model ~/VoxCPM1.5 --devices 0 --concurrency 1 --warmup 1 --iters 5 \
  --target-text-file benchmark/target_text_100w_en.txt
```

See `benchmark/README.md` for more flags.

### Reference Results (RTX 4090)

All reference numbers in this section are measured on NVIDIA GeForce RTX 4090.

The benchmark reports `RTF_per_req_mean`, defined as the mean over requests of
`(request_wall_time / request_audio_duration)` under the given concurrency.

Test setup:

- GPU: NVIDIA GeForce RTX 4090
- Model: `~/VoxCPM1.5`
- Benchmark: `benchmark/bench_inference.py`
- Runs: `--warmup 1 --iters 5`

Short prompt (`"Hello world."`):

Note: with a very short prompt, the model's stopping behavior can be noisy, so output audio duration (and thus RTF) may have high variance at higher concurrency.

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.1741 ± 0.0012 | 0.1741 ± 0.0012 | 0.1918 ± 0.0127 |
| 8 | 0.1804 ± 0.0041 | 0.1807 ± 0.0040 | 0.2353 ± 0.0162 |
| 16 | 0.1870 ± 0.0055 | 0.1878 ± 0.0054 | 0.3009 ± 0.0094 |
| 32 | 0.1924 ± 0.0052 | 0.1932 ± 0.0051 | 0.4055 ± 0.0099 |
| 64 | 0.2531 ± 0.0823 | 0.2918 ± 0.0938 | 0.6755 ± 0.0668 |

Long prompt (`benchmark/target_text_100w_en.txt`):

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.1909 ± 0.0102 | 0.1909 ± 0.0102 | 0.0805 ± 0.0007 |
| 8 | 0.1902 ± 0.0021 | 0.1905 ± 0.0021 | 0.1159 ± 0.0004 |
| 16 | 0.2044 ± 0.0050 | 0.2050 ± 0.0051 | 0.1825 ± 0.0007 |
| 32 | 0.2168 ± 0.0034 | 0.2185 ± 0.0032 | 0.3207 ± 0.0022 |
| 64 | 0.3235 ± 0.0063 | 0.3250 ± 0.0064 | 0.5556 ± 0.0033 |

Closed-loop users benchmark (`benchmark/bench_closed_loop_users.py`):

- Model: `~/VoxCPM1.5`
- Command:

```bash
uv run python benchmark/bench_closed_loop_users.py \
  --model ~/VoxCPM1.5 \
  --num-users 60 --warmup-s 5 --duration-s 60 \
  --target-text-file benchmark/target_text_100w_en.txt \
  --max-generate-length 2000
```

Results (measured window):

| item | value |
|---|---:|
| sample_rate (Hz) | 44100 |
| users | 60 |
| started | 119 |
| achieved rps | 1.98 |
| ok | 119 |
| err | 0 |

TTFB (seconds, ok requests):

| p50 | p90 | p95 | p99 | mean | stdev |
|---:|---:|---:|---:|---:|---:|
| 0.2634 | 0.3477 | 0.3531 | 0.3631 | 0.2884 | 0.0451 |

RTF (wall/audio, ok requests):

| p50 | p90 | p95 | p99 | mean | stdev |
|---:|---:|---:|---:|---:|---:|
| 0.7285 | 0.7946 | 0.8028 | 0.8255 | 0.6929 | 0.1062 |

## Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM)
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)

## License

MIT License

## Known Issue

If you see the errors below:
```
ValueError: Missing parameters: ['base_lm.embed_tokens.weight', 'base_lm.layers.0.self_attn.qkv_proj.weight', ... , 'stop_proj.weight', 'stop_proj.bias', 'stop_head.weight']
[rank0]:[W1106 07:26:04.469150505 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```

It's because nanovllm loads model parameters from `*.safetensors`, but some VoxCPM releases ship weights as `.pt`.

Fix:

- use a safetensors-converted checkpoint (or convert the checkpoint yourself)
- ensure the `*.safetensors` files live next to `config.json` in the model directory
