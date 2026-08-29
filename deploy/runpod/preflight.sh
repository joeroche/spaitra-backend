#!/usr/bin/env bash
set -euo pipefail

PERSIST_ROOT="${PERSIST_ROOT:-/workspace}"
RUN_USER="${RUN_USER:-spaitra}"
FAIL=0

ok() { echo "[ok]   $*"; }
warn() { echo "[warn] $*"; }
fail() { echo "[fail] $*"; FAIL=1; }

require_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    ok "command available: $1"
  else
    fail "missing command: $1"
  fi
}

echo "RunPod preflight"

for cmd in nvidia-smi python3 pip git ffmpeg rsync curl ssh; do
  require_cmd "$cmd"
done

if [[ -d "$PERSIST_ROOT" && -w "$PERSIST_ROOT" ]]; then
  ok "persistent root writable: $PERSIST_ROOT"
else
  fail "persistent root missing or not writable: $PERSIST_ROOT"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
fi

free -h
df -h "$PERSIST_ROOT" || true

if python3 -m venv --help >/dev/null 2>&1; then
  ok "python venv available"
else
  fail "python3 -m venv unavailable"
fi

if [[ -n "${RUNPOD_PUBLIC_IP:-}" ]]; then
  ok "RUNPOD_PUBLIC_IP present"
else
  warn "RUNPOD_PUBLIC_IP not set"
fi

if [[ -n "${RUNPOD_TCP_PORT_22:-}" ]]; then
  ok "RUNPOD_TCP_PORT_22 present"
else
  warn "RUNPOD_TCP_PORT_22 not set"
fi

if [[ -f "/opt/spaitra/.env" ]]; then
  ok "core env file present"
else
  warn "missing /opt/spaitra/.env"
fi

if [[ -f "/opt/spaitra/.ocr.env" ]]; then
  ok "ocr env file present"
else
  warn "missing /opt/spaitra/.ocr.env"
fi

HF_TOKEN_PATH="/opt/spaitra/cache/huggingface/token"
if [[ -f "$HF_TOKEN_PATH" ]]; then
  ok "huggingface token present"
elif [[ -n "${HF_TOKEN:-}" ]]; then
  ok "HF_TOKEN env present"
else
  warn "huggingface token missing"
fi

if command -v ollama >/dev/null 2>&1 || [[ -x /usr/local/bin/ollama ]]; then
  ok "ollama binary present"
else
  warn "ollama binary missing"
fi

if command -v srv.us >/dev/null 2>&1 || [[ -x /usr/local/bin/srv.us ]]; then
  ok "srv.us binary present"
else
  warn "srv.us binary missing"
fi

if id -u "$RUN_USER" >/dev/null 2>&1; then
  ok "service user present: $RUN_USER"
else
  warn "service user missing: $RUN_USER"
fi

exit "$FAIL"
