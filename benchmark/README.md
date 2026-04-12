# Benchmark

End-to-end inference benchmarking for VoxCPM.

## Run

```bash
uv run python benchmark/bench_inference.py --model ~/VoxCPM1.5 --concurrency 4 --iters 5 --warmup 1
```

Fixed-RPS TTFB (open-loop) for long-audio load:

```bash
uv run python benchmark/bench_open_loop_users.py --model ~/VoxCPM1.5 --rps 30 --duration-s 60 \
  --target-text-file benchmark/target_text_100w_en.txt --max-generate-length 2000
```

In in-process mode, the script also reports RTF (wall_time / generated_audio_seconds).

You can also benchmark the deployment service endpoint:

```bash
uv run python benchmark/bench_open_loop_users.py --url http://127.0.0.1:8000/generate --rps 30 --duration-s 60 \
  --target-text-file benchmark/target_text_100w_en.txt --max-generate-length 2000 --http-consume-full
```

Key flags:

- `--concurrency`: number of concurrent `generate()` requests
- `--max-generate-length`: maximum number of generation steps per request
- `--devices`: CUDA devices, e.g. `0` or `0,1`
- `--json-out`: write machine-readable results

Closed-loop "N users" benchmark (each user sends the next request immediately after the previous finishes):

```bash
uv run python benchmark/bench_closed_loop_users.py --model ~/VoxCPM1.5 --num-users 60 --duration-s 60 --warmup-s 5 \
  --target-text-file benchmark/target_text_100w_en.txt --max-generate-length 2000
```

## Notes

- Metrics are measured from the parent process wall time and include IPC overhead.
- If the model directory is local, the script reads `config.json` to infer `sample_rate` for RTF; otherwise provide `--sample-rate`.
- `RTF_per_req_mean` is computed as the average over requests of `(request_wall_time / request_audio_duration)`.
