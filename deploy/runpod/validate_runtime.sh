#!/usr/bin/env bash
set -euo pipefail

CONF="${CONF:-/etc/supervisor/spaitra-supervisord.conf}"
CORE_URL="${CORE_URL:-http://127.0.0.1:5000}"
OCR_URL="${OCR_URL:-http://127.0.0.1:8002/health}"
RUN_USER="${RUN_USER:-spaitra}"
FAIL=0

ok() { echo "[ok]   $*"; }
fail() { echo "[fail] $*"; FAIL=1; }

expect_run_user_rw_dir() {
  local dir="$1"
  local probe_name=".runpod_validate_probe.$$"
  if sudo -u "$RUN_USER" bash -lc "
    set -e
    cd '$dir'
    : > '$probe_name'
    rm -f '$probe_name'
  " >/dev/null 2>&1; then
    ok "service user can read/write: $dir"
  else
    fail "service user cannot read/write: $dir"
  fi
}

expect_file_mode() {
  local path="$1"
  local mode="$2"
  if [[ ! -e "$path" ]]; then
    fail "missing: $path"
    return
  fi
  local actual
  actual="$(stat -c '%a' "$path")"
  if [[ "$actual" == "$mode" ]]; then
    ok "mode $mode: $path"
  else
    fail "mode mismatch for $path: expected $mode got $actual"
  fi
}

expect_workspace_env_mode() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    fail "missing: $path"
    return
  fi
  local actual
  actual="$(stat -c '%a' "$path")"
  if [[ "$actual" == "600" ]]; then
    ok "mode 600: $path"
  elif [[ "$actual" == "666" ]]; then
    echo "[warn] workspace fuse mount ignored chmod for $path; using /etc/spaitra runtime copy for service reads"
  else
    fail "unexpected workspace env mode for $path: $actual"
  fi
}

if id -u "$RUN_USER" >/dev/null 2>&1; then
  ok "service user present: $RUN_USER"
else
  fail "service user missing: $RUN_USER"
fi

if [[ -L /opt/spaitra && "$(readlink /opt/spaitra)" == "/workspace/spaitra" ]]; then
  ok "/opt/spaitra symlink points at /workspace/spaitra"
else
  fail "/opt/spaitra is not the expected compatibility symlink"
fi

expect_workspace_env_mode /opt/spaitra/.env
expect_workspace_env_mode /opt/spaitra/.ocr.env
expect_file_mode /etc/spaitra/.env 600
expect_file_mode /etc/spaitra/.ocr.env 600

if sudo -u "$RUN_USER" test -r /etc/spaitra/.env && sudo -u "$RUN_USER" test -r /etc/spaitra/.ocr.env; then
  ok "service user can read runtime env files"
else
  fail "service user cannot read runtime env files"
fi

expect_run_user_rw_dir /opt/spaitra/backend-copy/data
expect_run_user_rw_dir /opt/spaitra/backend-copy/logs
if [[ -f /opt/spaitra/backend-copy/data/memory.db ]]; then
  if sudo -u "$RUN_USER" test -r /opt/spaitra/backend-copy/data/memory.db; then
    ok "service user can read database file"
  else
    fail "service user cannot read database file"
  fi
fi

for path in /opt/spaitra/venv-core/bin/python /opt/spaitra/venv-ocr/bin/python; do
  if [[ -x "$path" ]]; then
    ok "runtime env present: $path"
  else
    fail "missing runtime env: $path"
  fi
done

for path in /opt/spaitra/cache /opt/spaitra/cache/huggingface /opt/spaitra/cache/ollama; do
  if [[ -d "$path" ]]; then
    ok "cache path present: $path"
  else
    fail "missing cache path: $path"
  fi
done

if [[ -x /usr/bin/supervisorctl || -x /usr/local/bin/supervisorctl ]]; then
  if supervisorctl -c "$CONF" status >/tmp/spaitra_supervisor_status.txt 2>/dev/null; then
    ok "supervisor status readable"
    cat /tmp/spaitra_supervisor_status.txt
    python3 - <<'PY'
from pathlib import Path
env = {}
for line in Path("/opt/spaitra/.env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    env[key] = value

expected = {"spaitra-sshd", "ollama", "spaitra-ocr", "spaitra-core", "spaitra-tunnel"}
seen = set()
bad = []
for line in Path("/tmp/spaitra_supervisor_status.txt").read_text().splitlines():
    if not line.strip():
        continue
    name, status = line.split(None, 1)
    seen.add(name)
    enabled = True
    if name == "ollama":
        enabled = env.get("ENABLE_OLLAMA_SERVICE", "1") == "1"
    elif name == "spaitra-tunnel":
        enabled = env.get("ENABLE_SRVUS", "1") == "1"
    if enabled and "RUNNING" not in status:
        bad.append((name, status))
missing = expected - seen
if missing:
    raise SystemExit(f"missing supervisor programs: {sorted(missing)}")
if bad:
    raise SystemExit(f"non-running critical programs: {bad}")
print("supervisor_programs_ok")
PY
    ok "supervisor programs"
  else
    fail "supervisorctl could not read status"
  fi
else
  fail "supervisorctl missing"
fi

for spec in 127.0.0.1:5000 127.0.0.1:8002; do
  host="${spec%:*}"
  port="${spec##*:}"
  if ss -ltn | awk '{print $4}' | grep -q "${host}:${port}\$"; then
    ok "listener present: ${host}:${port}"
  else
    fail "listener missing: ${host}:${port}"
  fi
done

if curl -fsS "$CORE_URL/health" >/dev/null; then
  ok "core health"
else
  fail "core health failed"
fi

if curl -fsS "$OCR_URL" >/dev/null; then
  ok "ocr health"
else
  fail "ocr health failed"
fi

DEPS_HTTP="$(curl -sS -o /tmp/spaitra_deps.json -w "%{http_code}" "$CORE_URL/health/dependencies" || true)"
if [[ "$DEPS_HTTP" == "200" ]]; then
  python3 - <<'PY'
import json
payload = json.load(open("/tmp/spaitra_deps.json"))
deps = payload.get("dependencies", {})
assert deps.get("ffmpeg", {}).get("available"), "ffmpeg unavailable"
ocr = deps.get("ocr", {})
assert (not ocr.get("enabled")) or ocr.get("reachable"), "ocr unreachable"
print("dependency_health_ok")
PY
  ok "dependency health"
else
  fail "dependency health failed: $DEPS_HTTP"
fi

python3 - <<'PY'
from pathlib import Path
env = {}
for line in Path("/etc/spaitra/.env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    env[k] = v
required = {
    "ENABLE_DEPTH": "1",
    "ENABLE_OCR": "1",
    "SAVE_VRAM": "0",
    "STARTUP_WARM_MODE": "full",
    "ENABLE_CORE_SERVICE": "1",
}
for key, expected in required.items():
    actual = env.get(key)
    if actual != expected:
        raise SystemExit(f"{key} expected {expected!r} got {actual!r}")
print("core_env_expected_values_ok")
PY
ok "core env steady-state flags"

python3 - <<'PY'
from pathlib import Path
env = {}
for line in Path("/etc/spaitra/.ocr.env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    env[k] = v
if env.get("OCR_MAX_CONCURRENCY") != "4":
    raise SystemExit(f"OCR_MAX_CONCURRENCY expected '4' got {env.get('OCR_MAX_CONCURRENCY')!r}")
if env.get("OCR_USE_GPU") != "1":
    raise SystemExit(f"OCR_USE_GPU expected '1' got {env.get('OCR_USE_GPU')!r}")
if env.get("OCR_GPU_REQUIRED") != "1":
    raise SystemExit(f"OCR_GPU_REQUIRED expected '1' got {env.get('OCR_GPU_REQUIRED')!r}")
print("ocr_env_expected_values_ok")
PY
ok "ocr env steady-state flags"

python3 - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8002/health", timeout=30) as resp:
    payload = json.loads(resp.read().decode() or "{}")
ocr = payload.get("ocr", {})
assert ocr.get("gpu_requested") is True, f"gpu_requested={ocr.get('gpu_requested')!r}"
assert ocr.get("gpu_required") is True, f"gpu_required={ocr.get('gpu_required')!r}"
assert ocr.get("gpu_available") is True, f"gpu_available={ocr.get('gpu_available')!r}"
assert ocr.get("gpu_state") == "ready", f"gpu_state={ocr.get('gpu_state')!r}"
print("ocr_gpu_health_ok")
PY
ok "ocr gpu health"

exit "$FAIL"
