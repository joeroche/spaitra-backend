#!/usr/bin/env bash
set -euo pipefail

CORE_URL="${CORE_URL:-http://127.0.0.1:5000}"
OCR_HEALTH_URL="${OCR_HEALTH_URL:-http://127.0.0.1:8001/health}"
ENV_ROOT="${ENV_ROOT:-/opt/spaitra}"
API_KEY_VALUE="${API_KEY:-}"
TRANSCRIBE_AUDIO_PATH="${TRANSCRIBE_AUDIO_PATH:-}"

if [[ -z "$API_KEY_VALUE" ]] && [[ -f "$ENV_ROOT/.env" ]]; then
  API_KEY_VALUE="$(grep -E '^API_KEY=' "$ENV_ROOT/.env" | head -n 1 | cut -d= -f2- || true)"
fi

echo "Running post-cutover smoke checks"
echo "  core: $CORE_URL"
echo "  ocr:  $OCR_HEALTH_URL"

echo "[1/6] core health"
curl -fsS "$CORE_URL/health" >/dev/null

echo "[2/6] OCR health"
curl -fsS "$OCR_HEALTH_URL" >/dev/null

echo "[3/6] dependency health"
DEPS_HTTP="$(curl -sS -o /tmp/health_deps.out -w "%{http_code}" "$CORE_URL/health/dependencies")"
if [[ "$DEPS_HTTP" == "404" ]]; then
  echo "dependency_health_skipped: /health/dependencies not available on this build"
elif [[ "$DEPS_HTTP" != "200" ]]; then
  echo "dependency health check failed: /health/dependencies returned $DEPS_HTTP" >&2
  exit 1
else
  DEPS_JSON="$(cat /tmp/health_deps.out)"
  python3 - <<'PY' "$DEPS_JSON"
import json
import sys
payload = json.loads(sys.argv[1])
ffmpeg_ok = bool(payload.get("dependencies", {}).get("ffmpeg", {}).get("available"))
ocr = payload.get("dependencies", {}).get("ocr", {})
ocr_enabled = bool(ocr.get("enabled"))
ocr_ok = (not ocr_enabled) or bool(ocr.get("reachable"))
if not ffmpeg_ok:
    raise SystemExit("ffmpeg unavailable in /health/dependencies")
if not ocr_ok:
    raise SystemExit("ocr unreachable in /health/dependencies")
print("dependency_health_ok")
PY
fi

if [[ -z "$API_KEY_VALUE" ]]; then
  echo "[4/6] settings check skipped (API_KEY unavailable)"
  echo "[5/6] items count check skipped (API_KEY unavailable)"
else
  echo "[4/6] settings route"
  curl -fsS -H "X-API-Key: $API_KEY_VALUE" "$CORE_URL/settings" >/dev/null

  echo "[5/6] item count"
  ITEMS_JSON="$(curl -fsS -H "X-API-Key: $API_KEY_VALUE" "$CORE_URL/items")"
  python3 - <<'PY' "$ITEMS_JSON"
import json
import sys
payload = json.loads(sys.argv[1])
count = payload.get("count")
if count is None:
    raise SystemExit("/items missing count field")
print(f"items_count={count}")
PY
fi

if [[ -n "$TRANSCRIBE_AUDIO_PATH" ]]; then
  if [[ -z "$API_KEY_VALUE" ]]; then
    echo "[6/6] transcribe smoke skipped (API_KEY unavailable)"
  elif [[ ! -f "$TRANSCRIBE_AUDIO_PATH" ]]; then
    echo "[6/6] transcribe smoke skipped (file not found: $TRANSCRIBE_AUDIO_PATH)"
  else
    echo "[6/6] transcribe route"
    curl -fsS -X POST \
      -H "X-API-Key: $API_KEY_VALUE" \
      -F "audio=@$TRANSCRIBE_AUDIO_PATH" \
      "$CORE_URL/transcribe" >/dev/null
  fi
else
  echo "[6/6] transcribe smoke skipped (set TRANSCRIBE_AUDIO_PATH to enable)"
fi

echo "Smoke checks passed."
