#!/usr/bin/env python3
import argparse
import base64
import concurrent.futures as cf
import json
import statistics
import time
from pathlib import Path

import requests

PROMPT_TEXT_DEFAULT = "不会是说天然香就不会有焦油，万物只要能燃烧就会有烟雾和焦油，区别只是多还是少而已。"
TARGET_TEXT_DEFAULT = "今天天气不错，我们继续测试一下后续续写的稳定性。"


def run_once(base_url: str, hifi_id: str, target_text: str, timeout: tuple[int, int]):
    t0 = time.time()
    first = None
    size = 0
    try:
        with requests.post(
            base_url + "/generate",
            json={"target_text": target_text, "hifi_id": hifi_id},
            stream=True,
            timeout=timeout,
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
            "ok": True,
            "ttfb_sec": None if first is None else round(first - t0, 3),
            "total_sec": round(t1 - t0, 3),
            "bytes": size,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": repr(e),
            "ttfb_sec": None if first is None else round(first - t0, 3),
            "total_sec": round(time.time() - t0, 3),
            "bytes": size,
        }


def p95(vals):
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    vals = sorted(vals)
    k = max(0, min(len(vals) - 1, int(round(0.95 * (len(vals) - 1)))))
    return vals[k]


def summarize(results):
    oks = [r for r in results if r["ok"]]
    totals = [r["total_sec"] for r in oks]
    ttfbs = [r["ttfb_sec"] for r in oks if r["ttfb_sec"] is not None]
    return {
        "success": len(oks),
        "failed": len(results) - len(oks),
        "avg_ttfb_sec": None if not ttfbs else round(statistics.mean(ttfbs), 3),
        "avg_total_sec": None if not totals else round(statistics.mean(totals), 3),
        "p95_total_sec": None if not totals else round(p95(totals), 3),
        "min_total_sec": None if not totals else round(min(totals), 3),
        "max_total_sec": None if not totals else round(max(totals), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8800")
    ap.add_argument(
        "--ref-wav", default="/opt/nanovllm-voxcpm/tools/hifi_reference.wav"
    )
    ap.add_argument("--wav-format", default="wav")
    ap.add_argument("--prompt-text", default=PROMPT_TEXT_DEFAULT)
    ap.add_argument("--target-text", default=TARGET_TEXT_DEFAULT)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--connect-timeout", type=int, default=10)
    ap.add_argument("--read-timeout", type=int, default=120)
    ap.add_argument("--do-service-warmup", action="store_true", default=True)
    ap.add_argument("--sleep-after-add", type=float, default=3.0)
    ap.add_argument("--hifi-warmup-runs", type=int, default=2)
    ap.add_argument("--sleep-after-warmups", type=float, default=0.0)
    ap.add_argument("--sleep-between-groups", type=float, default=0.0)
    args = ap.parse_args()

    ref_path = Path(args.ref_wav)
    if not ref_path.exists():
        raise SystemExit(f"ref wav not found: {ref_path}")

    with open(ref_path, "rb") as f:
        wav_b64 = base64.b64encode(f.read()).decode("ascii")

    timeout = (args.connect_timeout, args.read_timeout)
    requests.get(args.base_url + "/health", timeout=10).raise_for_status()

    service_warmup = None
    if args.do_service_warmup:
        t0 = time.time()
        r = requests.post(
            args.base_url + "/generate_blocking_wav",
            json={"target_text": "你好", "max_generate_length": 128},
            timeout=120,
        )
        r.raise_for_status()
        service_warmup = {"sec": round(time.time() - t0, 3), "bytes": len(r.content)}

    add_t0 = time.time()
    add = requests.post(
        args.base_url + "/add_hifi",
        json={
            "wav_base64": wav_b64,
            "wav_format": args.wav_format,
            "prompt_text": args.prompt_text,
        },
        timeout=120,
    )
    add.raise_for_status()
    add_obj = add.json()
    hifi_id = add_obj["hifi_id"]
    add_sec = round(time.time() - add_t0, 3)

    try:
        if args.sleep_after_add > 0:
            time.sleep(args.sleep_after_add)
        hifi_warmups = [
            run_once(args.base_url, hifi_id, args.target_text, timeout)
            for _ in range(args.hifi_warmup_runs)
        ]
        if args.sleep_after_warmups > 0:
            time.sleep(args.sleep_after_warmups)

        out = {
            "hifi_id": hifi_id,
            "add_hifi_sec": add_sec,
            "service_warmup": service_warmup,
            "sleep_after_add_sec": args.sleep_after_add,
            "hifi_warmups": hifi_warmups,
            "sleep_after_warmups_sec": args.sleep_after_warmups,
            "sleep_between_groups_sec": args.sleep_between_groups,
            "prompt_text": args.prompt_text,
            "target_text": args.target_text,
            "ref_wav": str(ref_path),
            "runs": {},
        }
        for idx, n in enumerate(args.concurrency):
            with cf.ThreadPoolExecutor(max_workers=n) as ex:
                futs = [
                    ex.submit(
                        run_once, args.base_url, hifi_id, args.target_text, timeout
                    )
                    for _ in range(n)
                ]
                results = [f.result() for f in futs]
            out["runs"][str(n)] = {
                "summary": summarize(results),
                "results": results,
            }
            if idx != len(args.concurrency) - 1 and args.sleep_between_groups > 0:
                time.sleep(args.sleep_between_groups)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    finally:
        try:
            requests.delete(args.base_url + f"/hifi/{hifi_id}", timeout=30)
        except Exception:
            pass


if __name__ == "__main__":
    main()
