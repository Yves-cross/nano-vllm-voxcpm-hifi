#!/usr/bin/env python3
import argparse
import base64
import glob
import json
import os
import time
from typing import Any

import requests


def stream_generate(base_url: str, payload: dict[str, Any], timeout: tuple[int, int] = (10, 120)) -> dict[str, Any]:
    t0 = time.time()
    first = None
    size = 0
    with requests.post(base_url + '/generate', json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=16384):
            if not chunk:
                continue
            if first is None:
                first = time.time()
            size += len(chunk)
    t1 = time.time()
    return {
        'ttfb_sec': None if first is None else round(first - t0, 3),
        'total_sec': round(t1 - t0, 3),
        'bytes': size,
    }


def encode_latents(base_url: str, wav_b64: str, wav_format: str) -> tuple[str, float]:
    t0 = time.time()
    r = requests.post(base_url + '/encode_latents', json={'wav_base64': wav_b64, 'wav_format': wav_format}, timeout=120)
    r.raise_for_status()
    return r.json()['prompt_latents_base64'], round(time.time() - t0, 3)


def add_prompt(base_url: str, wav_b64: str, wav_format: str, prompt_text: str) -> tuple[str, float]:
    t0 = time.time()
    r = requests.post(base_url + '/add_prompt', json={
        'wav_base64': wav_b64,
        'wav_format': wav_format,
        'prompt_text': prompt_text,
    }, timeout=120)
    r.raise_for_status()
    return r.json()['prompt_id'], round(time.time() - t0, 3)


def delete_prompt(base_url: str, prompt_id: str) -> int:
    return requests.delete(base_url + f'/prompts/{prompt_id}', timeout=30).status_code


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Hot benchmark: prompt family (prompt_wav vs prompt_latents vs prompt_id)')
    p.add_argument('--base-url', default='http://127.0.0.1:8800')
    p.add_argument('--prompt-text', default='这是十五字并发测试语音样本文本')
    p.add_argument('--target-text', default='今天天气不错，我们继续测试一下后续续写的稳定性。')
    p.add_argument('--ref-wav', default=None, help='Prompt wav path; default=latest /tmp/gradio/*/audio.wav')
    p.add_argument('--wav-format', default='wav')
    p.add_argument('--runs', type=int, default=2)
    p.add_argument('--do-warmup', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ref = args.ref_wav
    if not ref:
        cands = sorted(glob.glob('/tmp/gradio/*/audio.wav'), key=os.path.getmtime, reverse=True)
        if not cands:
            raise SystemExit('no prompt wav found under /tmp/gradio/*/audio.wav; pass --ref-wav')
        ref = cands[0]

    with open(ref, 'rb') as f:
        wav_b64 = base64.b64encode(f.read()).decode('ascii')

    requests.get(args.base_url + '/health', timeout=10).raise_for_status()
    warmup_info = None
    if args.do_warmup:
        t0 = time.time()
        r = requests.post(args.base_url + '/generate_blocking_wav', json={'target_text': '你好', 'max_generate_length': 128}, timeout=120)
        r.raise_for_status()
        warmup_info = {'sec': round(time.time() - t0, 3), 'bytes': len(r.content)}

    out: dict[str, Any] = {
        'prompt_wav': ref,
        'prompt_text': args.prompt_text,
        'target_text': args.target_text,
        'runs': args.runs,
        'warmup': warmup_info,
        'routes': {},
    }

    payload_wav = {
        'target_text': args.target_text,
        'prompt_wav_base64': wav_b64,
        'prompt_wav_format': args.wav_format,
        'prompt_text': args.prompt_text,
    }
    wav_runs = [stream_generate(args.base_url, payload_wav) for _ in range(args.runs)]
    out['routes']['prompt_wav'] = {'runs': wav_runs}

    prompt_latents_b64, encode_sec = encode_latents(args.base_url, wav_b64, args.wav_format)
    payload_latents = {
        'target_text': args.target_text,
        'prompt_latents_base64': prompt_latents_b64,
        'prompt_text': args.prompt_text,
    }
    lat_runs = [stream_generate(args.base_url, payload_latents) for _ in range(args.runs)]
    out['routes']['prompt_latents'] = {'encode_sec': encode_sec, 'runs': lat_runs}

    prompt_id, add_prompt_sec = add_prompt(args.base_url, wav_b64, args.wav_format, args.prompt_text)
    try:
        payload_id = {
            'target_text': args.target_text,
            'prompt_id': prompt_id,
        }
        id_runs = [stream_generate(args.base_url, payload_id) for _ in range(args.runs)]
    finally:
        delete_status = delete_prompt(args.base_url, prompt_id)
    out['routes']['prompt_id'] = {'add_sec': add_prompt_sec, 'delete_status': delete_status, 'runs': id_runs}

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
