#!/usr/bin/env python3
import argparse
import base64
import concurrent.futures as cf
import glob
import json
import os
import statistics
import time

import requests

PROMPT_TEXT_DEFAULT = "不会是说天然香就不会有焦油，万物只要能燃烧就会有烟雾和焦油，区别只是多还是少而已。"
TARGET_TEXT_DEFAULT = "今天天气不错，我们继续测试一下后续续写的稳定性。"


def run_once(base_url: str, hifi_id: str, target_text: str, route: str, timeout: int):
    t0 = time.time()
    try:
        r = requests.post(
            base_url + route,
            json={"target_text": target_text, "hifi_id": hifi_id},
            timeout=timeout,
        )
        sec = time.time() - t0
        r.raise_for_status()
        return {"ok": True, "total_sec": round(sec, 3), "bytes": len(r.content)}
    except Exception as e:
        return {"ok": False, "total_sec": round(time.time() - t0, 3), "error": repr(e)}


def p95(vals):
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    vals = sorted(vals)
    k = max(0, min(len(vals) - 1, int(round(0.95 * (len(vals) - 1)))))
    return vals[k]


def summarize(results):
    oks = [r for r in results if r.get("ok")]
    totals = [r["total_sec"] for r in oks]
    return {
        "success": len(oks),
        "failed": len(results) - len(oks),
        "avg_total_sec": None if not totals else round(statistics.mean(totals), 3),
        "p95_total_sec": None if not totals else round(p95(totals), 3),
        "min_total_sec": None if not totals else round(min(totals), 3),
        "max_total_sec": None if not totals else round(max(totals), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8800")
    ap.add_argument(
        "--route",
        default="/generate_blocking",
        choices=["/generate_blocking", "/generate_blocking_wav"],
    )
    ap.add_argument("--ref-wav", default="/opt/nanovllm-voxcpm/tools/hifi_reference.wav")
    ap.add_argument("--wav-format", default="wav")
    ap.add_argument("--prompt-text", default=PROMPT_TEXT_DEFAULT)
    ap.add_argument("--target-text", default=TARGET_TEXT_DEFAULT)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--sleep-after-add", type=float, default=3.0)
    ap.add_argument("--warmup-runs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    ref = args.ref_wav
    if not os.path.exists(ref):
        cands = sorted(glob.glob("/tmp/gradio/*/audio.wav"), key=os.path.getmtime, reverse=True)
        if not cands:
            raise SystemExit("no reference wav found")
        ref = cands[0]
    with open(ref, "rb") as f:
        wav_b64 = base64.b64encode(f.read()).decode("ascii")

    requests.get(args.base_url + "/health", timeout=10).raise_for_status()
    requests.post(
        args.base_url + "/generate_blocking_wav",
        json={"target_text": "你好", "max_generate_length": 128},
        timeout=args.timeout,
    ).raise_for_status()

    add_t0 = time.time()
    r = requests.post(
        args.base_url + "/add_hifi",
        json={
            "wav_base64": wav_b64,
            "wav_format": args.wav_format,
            "prompt_text": args.prompt_text,
        },
        timeout=args.timeout,
    )
    r.raise_for_status()
    hifi_id = r.json()["hifi_id"]
    add_sec = round(time.time() - add_t0, 3)

    try:
        if args.sleep_after_add > 0:
            time.sleep(args.sleep_after_add)
        warmups = [
            run_once(args.base_url, hifi_id, args.target_text, args.route, args.timeout)
            for _ in range(args.warmup_runs)
        ]

        out = {
            "route": args.route,
            "hifi_id": hifi_id,
            "add_hifi_sec": add_sec,
            "sleep_after_add_sec": args.sleep_after_add,
            "warmups": warmups,
            "runs": {},
        }
        for n in args.concurrency:
            with cf.ThreadPoolExecutor(max_workers=n) as ex:
                futs = [
                    ex.submit(
                        run_once,
                        args.base_url,
                        hifi_id,
                        args.target_text,
                        args.route,
                        args.timeout,
                    )
                    for _ in range(n)
                ]
                results = [f.result() for f in futs]
            out["runs"][str(n)] = {"summary": summarize(results), "results": results}
        print(json.dumps(out, ensure_ascii=False, indent=2))
    finally:
        try:
            requests.delete(args.base_url + f"/hifi/{hifi_id}", timeout=30)
        except Exception:
            pass


if __name__ == "__main__":
    main()
