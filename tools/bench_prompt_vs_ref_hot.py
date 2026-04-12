#!/usr/bin/env python3
import argparse
import base64
import glob
import json
import os
import time
from typing import Any

import requests


def stream_generate(
    base_url: str, payload: dict[str, Any], timeout: tuple[int, int] = (10, 120)
) -> dict[str, Any]:
    t0 = time.time()
    first = None
    size = 0
    with requests.post(
        base_url + "/generate", json=payload, stream=True, timeout=timeout
    ) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=16384):
            if not chunk:
                continue
            if first is None:
                first = time.time()
            size += len(chunk)
    t1 = time.time()
    return {
        "ttfb_sec": None if first is None else round(first - t0, 3),
        "total_sec": round(t1 - t0, 3),
        "bytes": size,
    }


def add_reference(base_url: str, ref_b64: str, wav_format: str) -> tuple[str, float]:
    t0 = time.time()
    r = requests.post(
        base_url + "/add_reference",
        json={"wav_base64": ref_b64, "wav_format": wav_format},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["reference_id"], round(time.time() - t0, 3)


def delete_reference(base_url: str, reference_id: str) -> int:
    return requests.delete(
        base_url + f"/references/{reference_id}", timeout=30
    ).status_code


def add_prompt(
    base_url: str, ref_b64: str, wav_format: str, prompt_text: str
) -> tuple[str, float]:
    t0 = time.time()
    r = requests.post(
        base_url + "/add_prompt",
        json={
            "wav_base64": ref_b64,
            "wav_format": wav_format,
            "prompt_text": prompt_text,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["prompt_id"], round(time.time() - t0, 3)


def delete_prompt(base_url: str, prompt_id: str) -> int:
    return requests.delete(base_url + f"/prompts/{prompt_id}", timeout=30).status_code


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hot benchmark: prompt_id vs ref_audio_id")
    p.add_argument("--base-url", default="http://127.0.0.1:8800")
    p.add_argument("--text", default="这是十五字并发测试语音样本文本")
    p.add_argument(
        "--prompt-text",
        default=None,
        help="Prompt text for add_prompt; default=same as --text",
    )
    p.add_argument(
        "--ref-wav",
        default=None,
        help="Reference wav path; default=latest /tmp/gradio/*/audio.wav",
    )
    p.add_argument("--wav-format", default="wav")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument(
        "--do-warmup",
        action="store_true",
        help="Run a tiny generate_blocking_wav first to keep service hot",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prompt_text = args.prompt_text if args.prompt_text is not None else args.text

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

    requests.get(args.base_url + "/health", timeout=10).raise_for_status()

    warmup_info = None
    if args.do_warmup:
        t0 = time.time()
        r = requests.post(
            args.base_url + "/generate_blocking_wav",
            json={"target_text": "你好", "max_generate_length": 128},
            timeout=120,
        )
        r.raise_for_status()
        warmup_info = {"sec": round(time.time() - t0, 3), "bytes": len(r.content)}

    out: dict[str, Any] = {
        "reference_wav": ref,
        "text": args.text,
        "prompt_text": prompt_text,
        "runs": args.runs,
        "warmup": warmup_info,
        "routes": {},
    }

    ref_id, add_ref_sec = add_reference(args.base_url, ref_b64, args.wav_format)
    ref_runs = []
    try:
        for _ in range(args.runs):
            ref_runs.append(
                stream_generate(
                    args.base_url, {"target_text": args.text, "ref_audio_id": ref_id}
                )
            )
    finally:
        ref_delete = delete_reference(args.base_url, ref_id)
    out["routes"]["ref_audio_id"] = {
        "add_sec": add_ref_sec,
        "delete_status": ref_delete,
        "runs": ref_runs,
    }

    prompt_id, add_prompt_sec = add_prompt(
        args.base_url, ref_b64, args.wav_format, prompt_text
    )
    prompt_runs = []
    try:
        for _ in range(args.runs):
            prompt_runs.append(
                stream_generate(
                    args.base_url, {"target_text": args.text, "prompt_id": prompt_id}
                )
            )
    finally:
        prompt_delete = delete_prompt(args.base_url, prompt_id)
    out["routes"]["prompt_id"] = {
        "add_sec": add_prompt_sec,
        "delete_status": prompt_delete,
        "runs": prompt_runs,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
