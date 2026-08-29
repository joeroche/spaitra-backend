#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO="${SOURCE_REPO:-/opt/spaitra/TSA-soft-dev-backend-2026}"
TARGET_REPO="${TARGET_REPO:-/opt/spaitra/backend-copy}"
ENV_ROOT="${ENV_ROOT:-/opt/spaitra}"
RUN_AS_USER="${RUN_AS_USER:-spaitra}"
DRY_RUN=0
INCLUDE_DATA_DIR="${INCLUDE_DATA_DIR:-0}"
SKIP_VERIFY=0
SKIP_SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_REPO="$2"
      shift 2
      ;;
    --target)
      TARGET_REPO="$2"
      shift 2
      ;;
    --env-root)
      ENV_ROOT="$2"
      shift 2
      ;;
    --user)
      RUN_AS_USER="$2"
      shift 2
      ;;
    --include-data)
      INCLUDE_DATA_DIR=1
      shift
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    --skip-smoke)
      SKIP_SMOKE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if (( EUID != 0 )); then
  echo "Run as root (example: sudo bash deploy/cutover_backend_copy.sh)" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_REPO" ]]; then
  echo "Missing source repo: $SOURCE_REPO" >&2
  exit 1
fi

if [[ ! -d "$ENV_ROOT" ]]; then
  echo "Missing env root: $ENV_ROOT" >&2
  exit 1
fi

run_cmd() {
  echo "+ $*"
  if (( DRY_RUN == 0 )); then
    bash -lc "$*"
  fi
}

run_user_cmd() {
  local cmd="$1"
  echo "+ sudo -u $RUN_AS_USER -H bash -lc '$cmd'"
  if (( DRY_RUN == 0 )); then
    sudo -u "$RUN_AS_USER" -H bash -lc "$cmd"
  fi
}

REMOTE_URL="$(sudo -u "$RUN_AS_USER" -H bash -lc "cd '$SOURCE_REPO' && git remote get-url origin" 2>/dev/null || true)"
if [[ -z "$REMOTE_URL" ]]; then
  REMOTE_URL="https://github.com/tsa-softwaredev-26/TSA-soft-dev-backend-2026.git"
fi

echo "Cutover parameters"
echo "  source repo : $SOURCE_REPO"
echo "  target repo : $TARGET_REPO"
echo "  env root    : $ENV_ROOT"
echo "  run as user : $RUN_AS_USER"
echo "  include data: $INCLUDE_DATA_DIR"
echo "  dry run     : $DRY_RUN"
echo ""

echo "[1/9] Stop services"
run_cmd "systemctl stop spaitra-core spaitra-ocr"

echo "[2/9] Clone or refresh target repo"
run_user_cmd "cd '$ENV_ROOT' && if [[ ! -d '$TARGET_REPO/.git' ]]; then git clone '$REMOTE_URL' '$TARGET_REPO'; else cd '$TARGET_REPO' && git pull --ff-only; fi"

echo "[3/9] Sync gitignored runtime artifacts"
RUNTIME_DIRS=("checkpoints" "models" "logs" "benchmarks/images")
if [[ "$INCLUDE_DATA_DIR" == "1" ]]; then
  RUNTIME_DIRS+=("data")
fi
for rel in "${RUNTIME_DIRS[@]}"; do
  if [[ ! -d "$SOURCE_REPO/$rel" ]]; then
    echo "  [skip] source missing: $SOURCE_REPO/$rel"
    continue
  fi
  run_cmd "mkdir -p '$TARGET_REPO/$rel'"
  run_cmd "rsync -a '$SOURCE_REPO/$rel/' '$TARGET_REPO/$rel/'"
done

if (( SKIP_VERIFY == 0 )); then
  echo "[4/9] Verify source -> target artifacts"
  run_cmd "INCLUDE_DATA_DIR='$INCLUDE_DATA_DIR' bash '$TARGET_REPO/deploy/verify_cutover_data.sh' '$SOURCE_REPO' '$TARGET_REPO'"
else
  echo "[4/9] Verify source -> target artifacts (skipped)"
fi

echo "[5/9] Recreate virtualenvs and reinstall deps"
run_user_cmd "cd '$TARGET_REPO' && rm -rf venv-core venv-ocr && python3 -m venv venv-core && source venv-core/bin/activate && pip install --upgrade pip && pip install -e '.[core]' && deactivate && python3 -m venv venv-ocr && source venv-ocr/bin/activate && pip install --upgrade pip && pip install -e '.[ocr]' && deactivate"

echo "[6/9] Install/update systemd units"
run_cmd "bash '$TARGET_REPO/deploy/install.sh' '$TARGET_REPO' '$ENV_ROOT'"

echo "[7/9] Enable and start services"
run_cmd "systemctl enable --now spaitra-ocr spaitra-core"

if (( SKIP_SMOKE == 0 )); then
  echo "[8/9] Run smoke checks"
  run_cmd "ENV_ROOT='$ENV_ROOT' bash '$TARGET_REPO/deploy/smoke_backend_copy.sh'"
else
  echo "[8/9] Run smoke checks (skipped)"
fi

echo "[9/9] Done"
echo "Cutover complete. Keep '$SOURCE_REPO' for one release cycle as rollback safety."
