#!/usr/bin/env bash
# One-command benchmark runner. Runs the full benchmark, saves all artifacts
# to BASELINE_ROOT, and prints the baseline path.
#
# Usage:
#   ./scripts/run_benchmark.sh
#   BASELINE_ROOT=/opt/spaitra/accuracy_hardening_baselines ./scripts/run_benchmark.sh
#   ./scripts/run_benchmark.sh --no-depth --no-ocr   # pass flags to full_benchmark.py
#
# Must be run from the repository root with the core venv active.
# Tests run on the server: push changes to the server before running.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
SHA="$(git rev-parse --short HEAD)"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${SHA}"

if [[ -f /etc/spaitra/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/spaitra/.env
  set +a
fi

if [[ -d /opt/spaitra/cache/huggingface ]]; then
  export HF_HOME=/opt/spaitra/cache/huggingface
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
fi

DEFAULT_BASELINE_ROOT="${REPO_ROOT}/benchmarks/baselines/accuracy_hardening"
BASELINE_ROOT="${BASELINE_ROOT:-${DEFAULT_BASELINE_ROOT}}"

OUT_DIR="${BASELINE_ROOT}/${BRANCH}/${RUN_ID}"
mkdir -p "${OUT_DIR}"

echo "Branch:        ${BRANCH}"
echo "SHA:           ${SHA}"
echo "Run ID:        ${RUN_ID}"
echo "Baseline root: ${BASELINE_ROOT}"
echo "Output dir:    ${OUT_DIR}"
echo ""

# Save config snapshot (Settings + git context)
python -c "
from visual_memory.config import Settings
import json, subprocess, sys
s = Settings()
cfg = {f: getattr(s, f) for f in s.__dataclass_fields__}
cfg['git_sha'] = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
cfg['git_sha_short'] = '${SHA}'
cfg['branch'] = '${BRANCH}'
cfg['run_id'] = '${RUN_ID}'
cfg['worktree'] = subprocess.check_output(['git','rev-parse','--show-toplevel']).decode().strip()
cfg['baseline_root'] = '${BASELINE_ROOT}'
cfg['command'] = ' '.join(sys.argv)
json.dump(cfg, open('${OUT_DIR}/config.json', 'w'), indent=2, default=str)
print('Config snapshot saved.')
"

# Run benchmark
python -m visual_memory.benchmarks.full_benchmark \
    --dataset "${REPO_ROOT}/benchmarks/dataset.csv" \
    --images  "${REPO_ROOT}/benchmarks/images" \
    "$@"

# Copy primary results
cp "${REPO_ROOT}/benchmarks/results.csv"  "${OUT_DIR}/results.csv"
cp "${REPO_ROOT}/benchmarks/results.json" "${OUT_DIR}/results.json"

# Copy extended artifacts if present (produced by Step 0.2-0.4 extensions)
for f in inference_traces.csv failure_taxonomy.json top_k_analysis.json \
          confusion_pairs.json operating_points.csv fixed_fp_budget.json \
          debug_dump.txt; do
    src="${REPO_ROOT}/benchmarks/${f}"
    if [ -f "${src}" ]; then
        cp "${src}" "${OUT_DIR}/${f}"
    fi
done

# Generate metrics summary
python -c "
from visual_memory.benchmarks.baseline_writer import write_metrics_summary
metrics = write_metrics_summary('${OUT_DIR}')
import json
print('Metrics summary:')
print(json.dumps({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, indent=2))
"

# Update branch/latest symlink
LATEST="${BASELINE_ROOT}/${BRANCH}/latest"
[ -L "${LATEST}" ] && rm "${LATEST}"
ln -s "${RUN_ID}" "${LATEST}"

echo ""
echo "Baseline saved: ${OUT_DIR}"
echo ""
echo "To compare against frozen baseline:"
echo "  python -m visual_memory.benchmarks.baseline_writer compare \\"
echo "    ${BASELINE_ROOT}/main/frozen_baseline \\"
echo "    ${OUT_DIR}"
