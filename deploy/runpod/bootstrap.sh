#!/usr/bin/env bash
set -euo pipefail

if (( EUID != 0 )); then
  echo "Run as root." >&2
  exit 1
fi

PERSIST_ROOT="${PERSIST_ROOT:-/workspace/spaitra}"
COMPAT_ROOT="${COMPAT_ROOT:-/opt/spaitra}"
REPO_NAME="${REPO_NAME:-backend-copy}"
REPO_ROOT="${REPO_ROOT:-$COMPAT_ROOT/$REPO_NAME}"
VENV_CORE="${VENV_CORE:-$COMPAT_ROOT/venv-core}"
VENV_OCR="${VENV_OCR:-$COMPAT_ROOT/venv-ocr}"
CACHE_ROOT="${CACHE_ROOT:-$COMPAT_ROOT/cache}"
HF_HOME_DIR="${HF_HOME_DIR:-$CACHE_ROOT/huggingface}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$CACHE_ROOT/ollama}"
REPO_REMOTE="${REPO_REMOTE:-https://github.com/tsa-softwaredev-26/backend-copy.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
SKIP_GIT="${SKIP_GIT:-0}"
RUN_USER="${RUN_USER:-spaitra}"
RUN_GROUP="${RUN_GROUP:-spaitra}"
INSTALL_OLLAMA="${INSTALL_OLLAMA:-1}"
INSTALL_SRVUS="${INSTALL_SRVUS:-1}"
REBUILD_VENVS="${REBUILD_VENVS:-0}"
SETUP_WEIGHTS="${SETUP_WEIGHTS:-0}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || return 1
}

run_as_user() {
  if need_cmd sudo; then
    sudo -u "$RUN_USER" -H "$@"
  elif need_cmd runuser; then
    runuser -u "$RUN_USER" -- "$@"
  else
    su -s /bin/bash "$RUN_USER" -c "$(printf '%q ' "$@")"
  fi
}

ensure_apt_package() {
  local pkg="$1"
  dpkg -s "$pkg" >/dev/null 2>&1 && return 0
  DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg"
}

grant_spaitra_access() {
  local path="$1"
  [[ -e "$path" ]] || return 0
  if command -v setfacl >/dev/null 2>&1; then
    setfacl -m "u:${RUN_USER}:rwX" "$path" || true
    if [[ -d "$path" ]]; then
      setfacl -dm "u:${RUN_USER}:rwX" "$path" || true
    fi
  fi
}

user_can_rw_dir() {
  local dir="$1"
  run_as_user bash -lc "
    set -e
    test -d '$dir'
    probe=\"\$(
      cd '$dir' &&
      mktemp .runpod_rw_probe.XXXXXX
    )\"
    rm -f '$dir'/\"\$probe\"
  " >/dev/null 2>&1
}

repair_repo_data_dir() {
  local data_dir="$REPO_ROOT/data"
  local backup_dir=""
  local backup_db=""
  local temp_db=""
  local stamp
  local needs_repair=0
  local db_key=""
  local repair_python="python3"

  if [[ -x "$VENV_CORE/bin/python" ]]; then
    repair_python="$VENV_CORE/bin/python"
  fi
  if [[ -f /etc/spaitra/.env ]]; then
    db_key="$(
      python3 - <<'PY'
from pathlib import Path

value = ""
for line in Path("/etc/spaitra/.env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, raw = line.split("=", 1)
    if key == "DB_ENCRYPTION_KEY":
        value = raw
        break
print(value)
PY
    )"
  fi

  if ! user_can_rw_dir "$data_dir"; then
    needs_repair=1
  elif [[ -f "$data_dir/memory.db" ]] && ! run_as_user test -r "$data_dir/memory.db" >/dev/null 2>&1; then
    needs_repair=1
  elif [[ -n "$db_key" && -x "$repair_python" && -f "$data_dir/memory.db" ]]; then
    if ! run_as_user env DB_ENCRYPTION_KEY="$db_key" "$repair_python" - "$data_dir/memory.db" <<'PY' >/dev/null 2>&1
import os
import sys
from pysqlcipher3 import dbapi2 as sqlcipher

key = os.environ["DB_ENCRYPTION_KEY"]
path = sys.argv[1]
conn = sqlcipher.connect(path, check_same_thread=False)
escaped = key.replace("'", "''")
conn.execute(f"PRAGMA key = '{escaped}'")
conn.execute("PRAGMA kdf_iter = 500000")
conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
conn.close()
PY
    then
      needs_repair=1
    fi
  fi
  if [[ "$needs_repair" == "0" ]]; then
    return 0
  fi

  echo "repairing $data_dir for RunPod workspace compatibility"
  stamp="$(date +%Y%m%d_%H%M%S)"
  if [[ -e "$data_dir" ]]; then
    backup_dir="${data_dir}.rootbroken.${stamp}"
    mv "$data_dir" "$backup_dir"
    backup_db="$backup_dir/memory.db"
  fi

  run_as_user mkdir -p "$data_dir"

  if [[ -n "$backup_db" && -f "$backup_db" ]]; then
    temp_db="/tmp/runpod_memory_${stamp}.db"
    cp "$backup_db" "$temp_db"
    chown "$RUN_USER:$RUN_GROUP" "$temp_db"
    chmod 600 "$temp_db"
    run_as_user env DB_ENCRYPTION_KEY="$db_key" "$repair_python" - "$temp_db" "$data_dir/memory.db" <<'PY'
import os
import sqlite3
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)

key = os.environ.get("DB_ENCRYPTION_KEY", "").strip()
src_conn = sqlite3.connect(str(src))

if key:
    from pysqlcipher3 import dbapi2 as sqlcipher
    dst_conn = sqlcipher.connect(str(dst))
    escaped = key.replace("'", "''")
    dst_conn.execute(f"PRAGMA key = '{escaped}'")
    dst_conn.execute("PRAGMA kdf_iter = 500000")
else:
    dst_conn = sqlite3.connect(str(dst))

with src_conn, dst_conn:
    dst_conn.executescript("\n".join(src_conn.iterdump()))
    dst_conn.commit()
PY
    rm -f "$temp_db"
  fi

  if ! user_can_rw_dir "$data_dir"; then
    echo "failed to repair writable access for $data_dir" >&2
    exit 1
  fi
  if [[ -f "$data_dir/memory.db" ]] && ! run_as_user test -r "$data_dir/memory.db" >/dev/null 2>&1; then
    echo "repaired data dir exists but $RUN_USER still cannot read $data_dir/memory.db" >&2
    exit 1
  fi
}

echo "[1/9] Install base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update
for pkg in \
  supervisor rsync ffmpeg git curl openssh-server acl sudo \
  build-essential python3-dev python3.12-venv libsqlcipher-dev \
  libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
  libjpeg-dev libpng-dev libtiff-dev libwebp-dev; do
  ensure_apt_package "$pkg"
done

echo "[2/9] Prepare persistent root and compatibility symlink"
mkdir -p /workspace
mkdir -p "$PERSIST_ROOT"
mkdir -p \
  "$PERSIST_ROOT/logs/supervisor" \
  "$PERSIST_ROOT/run" \
  "$PERSIST_ROOT/migration" \
  "$HF_HOME_DIR" \
  "$OLLAMA_MODELS_DIR"
if id -u "$RUN_USER" >/dev/null 2>&1; then
  usermod -d "$PERSIST_ROOT" "$RUN_USER"
else
  groupadd -f "$RUN_GROUP"
  useradd --system --gid "$RUN_GROUP" --home-dir "$PERSIST_ROOT" --create-home --shell /bin/bash "$RUN_USER"
fi
if [[ -e "$COMPAT_ROOT" && ! -L "$COMPAT_ROOT" ]]; then
  mv "$COMPAT_ROOT" "${COMPAT_ROOT}.pre_runpod.$(date +%Y%m%d_%H%M%S)"
fi
ln -sfn "$PERSIST_ROOT" "$COMPAT_ROOT"
mkdir -p "$COMPAT_ROOT"
if ! chown "$RUN_USER:$RUN_GROUP" "$PERSIST_ROOT" 2>/dev/null; then
  echo "warning: top-level chown failed on $PERSIST_ROOT; falling back to ACL-based access"
fi
grant_spaitra_access "$PERSIST_ROOT"

echo "[3/9] Prepare sshd"
mkdir -p /run/sshd
ssh-keygen -A

echo "[4/9] Install optional ingress helpers"
if [[ "$INSTALL_SRVUS" == "1" ]] && ! need_cmd srv.us; then
  if ! curl -fsSL https://install.srv.us | bash; then
    echo "warning: srv.us installer unavailable; continue and install it later" >&2
  fi
fi
if [[ "$INSTALL_OLLAMA" == "1" ]] && [[ ! -x /usr/local/bin/ollama ]]; then
  if ! curl -fsSL https://ollama.com/install.sh | sh; then
    echo "warning: ollama installer unavailable; continue and install it later" >&2
  fi
fi

echo "[5/9] Clone or refresh repo"
if [[ "$SKIP_GIT" == "1" ]]; then
  if [[ ! -d "$REPO_ROOT" || ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    echo "SKIP_GIT=1 requires an existing repo checkout at $REPO_ROOT" >&2
    exit 1
  fi
elif [[ ! -d "$REPO_ROOT/.git" ]]; then
  run_as_user git clone --branch "$REPO_BRANCH" "$REPO_REMOTE" "$REPO_ROOT"
else
  run_as_user bash -lc "cd '$REPO_ROOT' && git fetch origin '$REPO_BRANCH' && git checkout '$REPO_BRANCH' && git pull --ff-only origin '$REPO_BRANCH'"
fi

echo "[6/9] Create default env files when missing"
if [[ ! -f "$COMPAT_ROOT/.env" ]]; then
  cp "$REPO_ROOT/deploy/env.example" "$COMPAT_ROOT/.env"
fi
if [[ ! -f "$COMPAT_ROOT/.ocr.env" ]]; then
  cp "$REPO_ROOT/deploy/ocr.env.example" "$COMPAT_ROOT/.ocr.env"
fi
chmod 600 "$COMPAT_ROOT/.env" "$COMPAT_ROOT/.ocr.env"
chown "$RUN_USER:$RUN_GROUP" "$COMPAT_ROOT/.env" "$COMPAT_ROOT/.ocr.env" 2>/dev/null || true
if command -v setfacl >/dev/null 2>&1; then
  setfacl -m "u:${RUN_USER}:rw" "$COMPAT_ROOT/.env" "$COMPAT_ROOT/.ocr.env" || true
fi
mkdir -p /etc/spaitra
install -o "$RUN_USER" -g "$RUN_GROUP" -m 600 "$COMPAT_ROOT/.env" /etc/spaitra/.env
install -o "$RUN_USER" -g "$RUN_GROUP" -m 600 "$COMPAT_ROOT/.ocr.env" /etc/spaitra/.ocr.env
install -d -o "$RUN_USER" -g "$RUN_GROUP" -m 700 /etc/spaitra/ssh
if [[ ! -f /etc/spaitra/ssh/id_ed25519 ]]; then
  run_as_user ssh-keygen -t ed25519 -N "" -f /etc/spaitra/ssh/id_ed25519
fi
chmod 600 /etc/spaitra/ssh/id_ed25519
chmod 644 /etc/spaitra/ssh/id_ed25519.pub

echo "[7/9] Build runtime environments"
if [[ "$REBUILD_VENVS" == "1" || ! -x "$VENV_CORE/bin/python" ]]; then
  rm -rf "$VENV_CORE"
  run_as_user bash -lc "python3 -m venv '$VENV_CORE' && source '$VENV_CORE/bin/activate' && export HF_HOME='$HF_HOME_DIR' && pip install --upgrade pip && cd '$REPO_ROOT' && pip install -e '.[core]' && bash deploy/install_gpu_deps.sh"
fi
if [[ "$REBUILD_VENVS" == "1" || ! -x "$VENV_OCR/bin/python" ]]; then
  rm -rf "$VENV_OCR"
  run_as_user bash -lc "python3 -m venv '$VENV_OCR' && source '$VENV_OCR/bin/activate' && pip install --upgrade pip && cd '$REPO_ROOT' && pip install -e '.[ocr]' && bash deploy/install_gpu_ocr_deps.sh"
fi
if [[ -e "$REPO_ROOT/venv-core" && ! -L "$REPO_ROOT/venv-core" ]]; then
  rm -rf "$REPO_ROOT/venv-core"
fi
if [[ -e "$REPO_ROOT/venv-ocr" && ! -L "$REPO_ROOT/venv-ocr" ]]; then
  rm -rf "$REPO_ROOT/venv-ocr"
fi
ln -sfn "$VENV_CORE" "$REPO_ROOT/venv-core"
ln -sfn "$VENV_OCR" "$REPO_ROOT/venv-ocr"
chown -h "$RUN_USER:$RUN_GROUP" "$REPO_ROOT/venv-core" "$REPO_ROOT/venv-ocr" 2>/dev/null || true

echo "[8/9] Optional model setup"
if [[ "$SETUP_WEIGHTS" == "1" ]]; then
  run_as_user bash -lc "export HF_HOME='$HF_HOME_DIR' OLLAMA_MODELS='$OLLAMA_MODELS_DIR'; cd '$REPO_ROOT' && source '$VENV_CORE/bin/activate' && python setup_weights.py"
fi
repair_repo_data_dir

echo "[9/9] Start supervisor-managed services"
install -m 0644 "$REPO_ROOT/deploy/runpod/supervisord.conf" /etc/supervisor/spaitra-supervisord.conf
chmod 700 "$COMPAT_ROOT/run"
chown -R "$RUN_USER:$RUN_GROUP" "$COMPAT_ROOT/logs" "$COMPAT_ROOT/run" 2>/dev/null || true
grant_spaitra_access "$COMPAT_ROOT/logs"
grant_spaitra_access "$COMPAT_ROOT/run"
if pgrep -x supervisord >/dev/null 2>&1; then
  supervisorctl -c /etc/supervisor/spaitra-supervisord.conf reread
  supervisorctl -c /etc/supervisor/spaitra-supervisord.conf update
  supervisorctl -c /etc/supervisor/spaitra-supervisord.conf restart all || true
else
  supervisord -c /etc/supervisor/spaitra-supervisord.conf
fi

echo "RunPod bootstrap complete."
