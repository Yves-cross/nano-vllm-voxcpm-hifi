#!/usr/bin/env bash
set -u

BASE_URL="${NANOVLLM_WARMUP_BASE_URL:-http://127.0.0.1:8800}"
MODE="${NANOVLLM_WARMUP_MODE:-zero}"
ZERO_TEXT="${NANOVLLM_WARMUP_TEXT:-你好}"
ZERO_MAX_LEN="${NANOVLLM_WARMUP_MAX_GENERATE_LENGTH:-128}"
HIFI_WAV_PATH="${NANOVLLM_HIFI_WARMUP_WAV_PATH:-}"
HIFI_PROMPT_TEXT="${NANOVLLM_HIFI_WARMUP_PROMPT_TEXT:-}"
HIFI_TARGET_TEXT="${NANOVLLM_HIFI_WARMUP_TARGET_TEXT:-今天天气不错，我们继续测试一下后续续写的稳定性。}"

for i in $(seq 1 60); do
  if curl -fsS --max-time 2 "$BASE_URL/health" >/dev/null 2>/dev/null; then
    echo post_start_warmup ready
    break
  fi
  sleep 1
done

echo "post_start_warmup begin mode=$MODE"

if [ "$MODE" = "hifi" ] && [ -n "$HIFI_WAV_PATH" ] && [ -n "$HIFI_PROMPT_TEXT" ] && [ -f "$HIFI_WAV_PATH" ]; then
  export BASE_URL HIFI_WAV_PATH HIFI_PROMPT_TEXT HIFI_TARGET_TEXT
  if python3 - <<'PY'
import base64, json, os, sys, urllib.request

base = os.environ['BASE_URL']
wav_path = os.environ['HIFI_WAV_PATH']
prompt_text = os.environ['HIFI_PROMPT_TEXT']
target_text = os.environ['HIFI_TARGET_TEXT']

with open(wav_path, 'rb') as f:
    wav_b64 = base64.b64encode(f.read()).decode('ascii')

def post_json(path, payload, timeout=120):
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(base + path, data=data, headers={'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8')

def delete(path, timeout=30):
    req = urllib.request.Request(base + path, method='DELETE')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status

obj = json.loads(post_json('/add_hifi', {
    'wav_base64': wav_b64,
    'wav_format': 'wav',
    'prompt_text': prompt_text,
}))
hifi_id = obj['hifi_id']
try:
    req = urllib.request.Request(base + '/generate_blocking_wav', data=json.dumps({
        'target_text': target_text,
        'hifi_id': hifi_id,
    }, ensure_ascii=False).encode('utf-8'), headers={'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        _ = r.read()
finally:
    try:
        delete('/hifi/' + hifi_id)
    except Exception:
        pass
print('ok')
PY
  then
    echo "post_start_warmup ok mode=hifi"
  else
    echo "post_start_warmup failed mode=hifi"
  fi
else
  if curl -fsS --max-time 120 -X POST \
    -H 'content-type: application/json' \
    --data "{\"target_text\":\"$ZERO_TEXT\",\"max_generate_length\":$ZERO_MAX_LEN}" \
    "$BASE_URL/generate_blocking_wav" \
    >/dev/null 2>/dev/null; then
    echo "post_start_warmup ok mode=zero"
  else
    echo "post_start_warmup failed mode=zero"
  fi
fi

exit 0
