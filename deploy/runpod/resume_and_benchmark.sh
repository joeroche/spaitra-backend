#!/usr/bin/env bash
# Idempotent RunPod resume script for the persistent /workspace deployment.
#
# Run on the pod as root from the repo root:
#   RUN_BENCHMARK=1 bash deploy/runpod/resume_and_benchmark.sh
#
# The script avoids expensive service warmup while repairing/downloads run, checks
# only the assets that are normally lost with the container overlay, validates the
# runtime, and optionally starts the frozen benchmark in the background.
set -euo pipefail

PERSIST_ROOT="${PERSIST_ROOT:-/workspace/spaitra}"
COMPAT_ROOT="${COMPAT_ROOT:-/opt/spaitra}"
REPO_ROOT="${REPO_ROOT:-$COMPAT_ROOT/backend-copy}"
CONF="${CONF:-/etc/supervisor/spaitra-supervisord.conf}"
VENV_CORE="${VENV_CORE:-$COMPAT_ROOT/venv-core}"
VENV_OCR="${VENV_OCR:-$COMPAT_ROOT/venv-ocr}"
HF_HOME_DIR="${HF_HOME_DIR:-$COMPAT_ROOT/cache/huggingface}"
BASELINE_ROOT="${BASELINE_ROOT:-$COMPAT_ROOT/accuracy_hardening_baselines}"
RUN_BENCHMARK="${RUN_BENCHMARK:-0}"
BENCHMARK_ARGS="${BENCHMARK_ARGS:-}"
REINSTALL_OCR="${REINSTALL_OCR:-auto}"
SETUP_WEIGHTS="${SETUP_WEIGHTS:-0}"
REBUILD_VENVS="${REBUILD_VENVS:-0}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-auto}"
START_CORE="${START_CORE:-auto}"
VALIDATE_RUNTIME="${VALIDATE_RUNTIME:-auto}"
LOG_DIR="${LOG_DIR:-$COMPAT_ROOT/logs/runpod}"
STATUS_FILE="${STATUS_FILE:-$LOG_DIR/resume_status.env}"

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

run_low_priority() {
  if command -v ionice >/dev/null 2>&1; then
    ionice -c2 -n7 nice -n 10 "$@"
  else
    nice -n 10 "$@"
  fi
}

supervisor() {
  supervisorctl -c "$CONF" "$@"
}

write_status() {
  {
    printf 'RUNPOD_RESUME_UPDATED=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'REPO_HEAD=%q\n' "$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || true)"
    printf 'BENCHMARK_PID=%q\n' "${BENCHMARK_PID:-}"
    printf 'BENCHMARK_LOG=%q\n' "${BENCHMARK_LOG:-}"
  } > "$STATUS_FILE"
}

require_root() {
  if (( EUID != 0 )); then
    echo "Run as root." >&2
    exit 1
  fi
}

stop_expensive_services() {
  if [[ -f "$CONF" ]] && pgrep -x supervisord >/dev/null 2>&1; then
    log "Stopping core and OCR while repairing runtime."
    supervisor stop spaitra-core spaitra-ocr >/dev/null 2>&1 || true
  fi
}

bootstrap_runtime() {
  if [[ "$SKIP_BOOTSTRAP" == "1" ]]; then
    log "Skipping bootstrap by request."
    return
  fi
  if [[ "$SKIP_BOOTSTRAP" == "auto" && -x "$VENV_CORE/bin/python" && -x "$VENV_OCR/bin/python" && -f "$CONF" ]]; then
    log "Runtime already exists; skipping bootstrap fast path."
    return
  fi
  log "Running idempotent bootstrap."
  cd "$REPO_ROOT"
  SKIP_GIT=1 \
  REBUILD_VENVS="$REBUILD_VENVS" \
  SETUP_WEIGHTS="$SETUP_WEIGHTS" \
  bash deploy/runpod/bootstrap.sh
}

ocr_import_ok() {
  "$VENV_OCR/bin/python" - <<'PY' >/dev/null 2>&1
import uvicorn
raise SystemExit(0 if hasattr(uvicorn, "run") else 1)
PY
}

repair_ocr_env() {
  if [[ "$REINSTALL_OCR" == "0" ]]; then
    return
  fi
  if [[ "$REINSTALL_OCR" == "auto" ]] && ocr_import_ok; then
    log "OCR uvicorn import is healthy."
    return
  fi

  log "Repairing OCR virtualenv packages."
  cd "$REPO_ROOT"
  run_low_priority bash -lc "
    set -euo pipefail
    source '$VENV_OCR/bin/activate'
    python -m pip install --upgrade pip
    pip install -e '.[ocr]'
    bash deploy/install_gpu_ocr_deps.sh
  "
}

ensure_depth_pro() {
  local dest="$REPO_ROOT/checkpoints/depth_pro.pt"
  local persistent_candidates=(
    "$PERSIST_ROOT/checkpoints/depth_pro.pt"
    "$COMPAT_ROOT/checkpoints/depth_pro.pt"
    "$PERSIST_ROOT/model_weights/depth_pro.pt"
    "$PERSIST_ROOT/cache/checkpoints/depth_pro.pt"
  )
  if [[ -s "$dest" ]]; then
    log "Depth Pro checkpoint present."
    return
  fi
  mkdir -p "$REPO_ROOT/checkpoints"
  for src in "${persistent_candidates[@]}"; do
    if [[ -s "$src" ]]; then
      log "Using persistent Depth Pro checkpoint: $src"
      if [[ "$src" != "$dest" ]]; then
        ln -sfn "$src" "$dest"
      fi
      return
    fi
  done

  log "Downloading only the missing Depth Pro checkpoint."
  cd "$REPO_ROOT"
  run_low_priority env \
    HF_HOME="$HF_HOME_DIR" \
    HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR/hub" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "$VENV_CORE/bin/python" - <<'PY'
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

repo_root = Path("/opt/spaitra/backend-copy")
dest = repo_root / "checkpoints" / "depth_pro.pt"
dest.parent.mkdir(parents=True, exist_ok=True)
path = Path(hf_hub_download("apple/DepthPro", filename="depth_pro.pt"))
if path.resolve() != dest.resolve():
    shutil.copy2(path, dest)
print(dest)
PY
}

check_benchmark_images() {
  local count
  count="$(find "$REPO_ROOT/benchmarks/images" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')"
  if (( count < 120 )); then
    echo "Benchmark images missing: found $count, expected at least 120." >&2
    echo "Populate $REPO_ROOT/benchmarks/images from the persistent volume or legacy rsync before benchmarking." >&2
    exit 1
  fi
  log "Benchmark images present: $count files."
}

start_and_validate() {
  local start_core="$START_CORE"
  local validate_runtime="$VALIDATE_RUNTIME"
  if [[ "$start_core" == "auto" ]]; then
    if [[ "$RUN_BENCHMARK" == "1" ]]; then
      start_core=0
    else
      start_core=1
    fi
  fi
  if [[ "$validate_runtime" == "auto" ]]; then
    if [[ "$RUN_BENCHMARK" == "1" ]]; then
      validate_runtime=quick
    else
      validate_runtime=full
    fi
  fi

  log "Starting OCR."
  supervisor start spaitra-ocr >/dev/null || true
  sleep 3
  if [[ "$start_core" == "1" ]]; then
    log "Starting core."
    supervisor start spaitra-core >/dev/null || true
  else
    log "Leaving core stopped for benchmark mode."
    supervisor stop spaitra-core >/dev/null 2>&1 || true
  fi

  case "$validate_runtime" in
    full)
      log "Running full runtime validation."
      cd "$REPO_ROOT"
      bash deploy/runpod/validate_runtime.sh
      ;;
    quick)
      log "Running quick benchmark-mode validation."
      log "Waiting for OCR service to become healthy (up to 30s)."
      for _i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        curl -fsS http://127.0.0.1:8002/health >/dev/null 2>&1 && break || true
        sleep 2
      done
      curl -fsS http://127.0.0.1:8002/health >/dev/null
      test -s "$REPO_ROOT/checkpoints/depth_pro.pt"
      ;;
    0|none|skip)
      log "Skipping runtime validation by request."
      ;;
    *)
      echo "Unknown VALIDATE_RUNTIME value: $validate_runtime" >&2
      exit 1
      ;;
  esac
}

warm_ocr_gpu() {
  # Send a real inference POST so the GPU CUDA context is initialized before the
  # benchmark starts. /health is a static import check and does not trigger inference.
  # Cold-start GPU init takes ~20s; subsequent calls are ~60ms.
  if [[ "$RUN_BENCHMARK" != "1" ]]; then
    return
  fi
  local warmup_img
  warmup_img="$(find "$REPO_ROOT/benchmarks/images/" -maxdepth 1 -type f \
    \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) 2>/dev/null | head -1)"
  if [[ -z "$warmup_img" ]]; then
    log "No benchmark images available for OCR warmup; skipping."
    return
  fi
  log "Warming OCR GPU via real inference call (expect up to 30s for first call)."
  local deadline=$(( $(date +%s) + 60 ))
  until curl -fsS http://127.0.0.1:8002/health >/dev/null 2>&1; do
    if (( $(date +%s) >= deadline )); then
      log "OCR service did not become healthy; skipping warmup."
      return
    fi
    sleep 2
  done
  if curl -fsS --max-time 45 -F "image=@${warmup_img}" http://127.0.0.1:8002/ocr >/dev/null 2>&1; then
    log "OCR GPU warmed successfully."
  else
    log "OCR warmup call failed or timed out; benchmark will cold-start OCR on first image."
  fi
}

start_benchmark() {
  if [[ "$RUN_BENCHMARK" != "1" ]]; then
    log "RUN_BENCHMARK is not 1; leaving benchmark start to operator."
    return
  fi

  BENCHMARK_LOG="${BENCHMARK_LOG:-/tmp/spaitra_benchmark_$(date +%Y%m%d_%H%M%S).log}"
  BENCHMARK_THREADS="${BENCHMARK_THREADS:-4}"
  log "Starting benchmark in background: $BENCHMARK_LOG"
  cd "$REPO_ROOT"
  # shellcheck disable=SC2086
  nohup bash -lc "
    set -euo pipefail
    source '$VENV_CORE/bin/activate'
    export BASELINE_ROOT='$BASELINE_ROOT'
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS='$BENCHMARK_THREADS'
    export MKL_NUM_THREADS='$BENCHMARK_THREADS'
    export OPENBLAS_NUM_THREADS='$BENCHMARK_THREADS'
    export NUMEXPR_NUM_THREADS='$BENCHMARK_THREADS'
    export VECLIB_MAXIMUM_THREADS='$BENCHMARK_THREADS'
    bash scripts/run_benchmark.sh $BENCHMARK_ARGS
  " > "$BENCHMARK_LOG" 2>&1 &
  BENCHMARK_PID="$!"
  write_status
  log "Benchmark pid: $BENCHMARK_PID"
}

main() {
  require_root
  log "RunPod resume started."
  stop_expensive_services
  bootstrap_runtime
  repair_ocr_env
  ensure_depth_pro
  check_benchmark_images
  start_and_validate
  warm_ocr_gpu
  start_benchmark
  write_status
  log "RunPod resume complete. Status: $STATUS_FILE"
}

main "$@"
