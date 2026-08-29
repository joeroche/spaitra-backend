#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO="${1:-/opt/spaitra/TSA-soft-dev-backend-2026}"
TARGET_REPO="${2:-/opt/spaitra/backend-copy}"

# Purpose: compare gitignored runtime artifacts between source and target repos
# before service restart. Reports files that exist in source but not in target.
RUNTIME_DIRS=(
  "checkpoints"
  "models"
  "logs"
  "benchmarks/images"
)

if [[ "${INCLUDE_DATA_DIR:-0}" == "1" ]]; then
  RUNTIME_DIRS+=("data")
fi

if [[ ! -d "$SOURCE_REPO" ]]; then
  echo "Missing source repo: $SOURCE_REPO" >&2
  exit 1
fi

if [[ ! -d "$TARGET_REPO" ]]; then
  echo "Missing target repo: $TARGET_REPO" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required for verification" >&2
  exit 1
fi

missing_total=0

echo "Verifying source -> target gitignored artifacts"
echo "  source: $SOURCE_REPO"
echo "  target: $TARGET_REPO"
echo ""

for rel in "${RUNTIME_DIRS[@]}"; do
  src="$SOURCE_REPO/$rel"
  dst="$TARGET_REPO/$rel"

  if [[ ! -d "$src" ]]; then
    echo "[SKIP] $rel (not present in source)"
    continue
  fi

  mkdir -p "$dst"

  # Show files present in source but missing in target.
  missing_lines="$(rsync -ain --ignore-existing --out-format='%n' "$src/" "$dst/" | sed '/\/$/d' || true)"
  count="$(printf '%s\n' "$missing_lines" | sed '/^$/d' | wc -l | tr -d ' ')"

  if [[ "$count" == "0" ]]; then
    echo "[OK]   $rel"
    continue
  fi

  missing_total=$((missing_total + count))
  echo "[MISS] $rel: $count file(s) missing in target"
  printf '%s\n' "$missing_lines" | sed '/^$/d' | head -n 20 | sed 's/^/       - /'
  if (( count > 20 )); then
    echo "       - ..."
  fi

done

echo ""
if (( missing_total > 0 )); then
  echo "Verification failed: $missing_total source file(s) are missing in target." >&2
  echo "Run the cutover sync step again before restarting services." >&2
  exit 1
fi

echo "Verification passed: no missing source files in selected runtime dirs."
