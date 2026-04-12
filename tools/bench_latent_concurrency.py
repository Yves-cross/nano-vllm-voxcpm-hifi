#!/usr/bin/env python3
import argparse
import base64
import concurrent.futures
import glob
import json
import os
import statistics
import subprocess
import tempfile
import threading
import time
from typing import Any

import requests

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 120


def ffprobe_duration(path: str):
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            text=True,
        ).strip()
        return float(out)
    except Exception:
        return None


def run_cmd(cmd):
    return subprocess.check_output(cmd, text=True).strip()


def gpu_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {"gpu": None, "processes": []}
    try:
        out = run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
        if out:
            mem, util = [x.strip() for x in out[0].split(",")[:2]]
            snap["gpu"] = {"memory_used_mib": float(mem), "util_gpu_pct": float(util)}
    except Exception as e:
        snap["gpu_error"] = repr(e)
    try:
        out = run_cmd(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
        for line in out:
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3:
                snap["processes"].append(
                    {
                        "pid": int(parts[0]),
                        "process_name": parts[1],
                        "used_gpu_memory_mib": float(parts[2]),
                    }
                )
    except Exception as e:
        snap["proc_error"] = repr(e)
    return snap


class GpuMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = run_cmd(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ]
                ).splitlines()
                if out:
                    mem, util = [x.strip() for x in out[0].split(",")[:2]]
                    self.samples.append(
                        {
                            "ts": time.time(),
                            "memory_used_mib": float(mem),
                            "util_gpu_pct": float(util),
                        }
                    )
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)


def percentile(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def encode_latents(base_url: str, ref_b64: str, wav_format: str):
    t0 = time.time()
    r = requests.post(
        base_url + "/encode_latents",
        json={"wav_base64": ref_b64, "wav_format": wav_format},
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    r.raise_for_status()
    return time.time() - t0, r.json()


def one_request(idx: int, base_url: str, payload: dict, barrier: threading.Barrier):
    barrier.wait()
    t0 = time.time()
    first = None
    total_bytes = 0
    try:
        with requests.post(
            base_url + "/generate",
            json=payload,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                if first is None:
                    first = time.time()
                total_bytes += len(chunk)
        t1 = time.time()
        return {
            "ok": True,
            "idx": idx,
            "ttfb_sec": None if first is None else round(first - t0, 3),
            "total_sec": round(t1 - t0, 3),
            "bytes": total_bytes,
        }
    except Exception as e:
        t1 = time.time()
        return {
            "ok": False,
            "idx": idx,
            "ttfb_sec": None if first is None else round(first - t0, 3),
            "total_sec": round(t1 - t0, 3),
            "bytes": total_bytes,
            "error": repr(e),
        }


def warmup_and_duration(base_url: str, payload: dict):
    fd, path = tempfile.mkstemp(suffix=".warmup.mp3")
    os.close(fd)
    t0 = time.time()
    first = None
    with requests.post(
        base_url + "/generate",
        json=payload,
        stream=True,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    ) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                if first is None:
                    first = time.time()
                f.write(chunk)
    t1 = time.time()
    dur = ffprobe_duration(path)
    return {
        "ttfb_sec": None if first is None else round(first - t0, 3),
        "total_sec": round(t1 - t0, 3),
        "output_duration_sec": dur,
        "rtf": None if not dur else round((t1 - t0) / dur, 3),
        "output_file": path,
    }


def summarize(name, results, batch_wall, baseline_duration, samples, before_gpu):
    oks = [r for r in results if r.get("ok")]
    errs = [r for r in results if not r.get("ok")]
    ttfbs = [r["ttfb_sec"] for r in oks if r.get("ttfb_sec") is not None]
    totals = [r["total_sec"] for r in oks]
    peak_mem = max((s["memory_used_mib"] for s in samples), default=None)
    peak_util = max((s["util_gpu_pct"] for s in samples), default=None)
    return {
        "name": name,
        "gpu_before_round": before_gpu,
        "batch_wall_sec": round(batch_wall, 3),
        "success_count": len(oks),
        "error_count": len(errs),
        "avg_ttfb_sec": None if not ttfbs else round(statistics.mean(ttfbs), 3),
        "p50_ttfb_sec": None if not ttfbs else round(percentile(ttfbs, 0.5), 3),
        "p95_ttfb_sec": None if not ttfbs else round(percentile(ttfbs, 0.95), 3),
        "avg_total_sec": None if not totals else round(statistics.mean(totals), 3),
        "p50_total_sec": None if not totals else round(percentile(totals, 0.5), 3),
        "p95_total_sec": None if not totals else round(percentile(totals, 0.95), 3),
        "approx_avg_rtf": None
        if not totals or not baseline_duration
        else round(statistics.mean(totals) / baseline_duration, 3),
        "approx_p95_rtf": None
        if not totals or not baseline_duration
        else round(percentile(totals, 0.95) / baseline_duration, 3),
        "peak_gpu_mem_mib": peak_mem,
        "peak_gpu_util_pct": peak_util,
        "first_error": errs[0].get("error") if errs else None,
        "error_examples": errs[:3],
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark Nano-vLLM latent-ref concurrency."
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8800")
    p.add_argument("--text", default="这是十五字并发测试语音样本文本")
    p.add_argument("--concurrency", nargs="+", type=int, default=[5, 10])
    p.add_argument(
        "--ref-wav",
        default=None,
        help="Path to reference wav. Default: latest /tmp/gradio/*/audio.wav",
    )
    p.add_argument("--wav-format", default="wav")
    return p.parse_args()


def main():
    args = parse_args()
    ref = args.ref_wav
    if not ref:
        cands = sorted(
            glob.glob("/tmp/gradio/*/audio.wav"), key=os.path.getmtime, reverse=True
        )
        if not cands:
            raise SystemExit(
                "no reference wav found under /tmp/gradio/*/audio.wav; pass --ref-wav"
            )
        ref = cands[0]
    with open(ref, "rb") as f:
        ref_b64 = base64.b64encode(f.read()).decode("ascii")

    requests.get(
        args.base_url + "/health", timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
    ).raise_for_status()
    enc_sec, enc = encode_latents(args.base_url, ref_b64, args.wav_format)
    payload = {
        "target_text": args.text,
        "ref_audio_latents_base64": enc["prompt_latents_base64"],
    }

    print(
        json.dumps(
            {
                "stage": "setup",
                "text": args.text,
                "text_len": len(args.text),
                "reference_wav": ref,
                "reference_duration_sec": ffprobe_duration(ref),
                "gpu_before": gpu_snapshot(),
                "encode_latents_sec": round(enc_sec, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    warm = warmup_and_duration(args.base_url, payload)
    print(json.dumps({"stage": "warmup", **warm}, ensure_ascii=False), flush=True)
    baseline_duration = warm.get("output_duration_sec") or 1.0

    rounds = []
    for c in args.concurrency:
        before = gpu_snapshot()
        mon = GpuMonitor(1.0)
        mon.start()
        barrier = threading.Barrier(c)
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
            futs = [
                ex.submit(one_request, i, args.base_url, payload, barrier)
                for i in range(c)
            ]
            results = [f.result() for f in futs]
        batch_wall = time.time() - t0
        mon.stop()
        summary = summarize(
            f"latent_concurrency_{c}",
            results,
            batch_wall,
            baseline_duration,
            mon.samples,
            before,
        )
        rounds.append(summary)
        print(json.dumps({"stage": "round", **summary}, ensure_ascii=False), flush=True)
        time.sleep(3)

    print(
        json.dumps(
            {"stage": "done", "gpu_after": gpu_snapshot(), "rounds": rounds},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
