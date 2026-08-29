#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST="${MANIFEST:-${REPO_ROOT}/benchmarks/hard_cases_manifest.json}"
RESULTS_JSON="${RESULTS_JSON:-${REPO_ROOT}/benchmarks/results.json}"
TRACES_CSV="${TRACES_CSV:-${REPO_ROOT}/benchmarks/inference_traces.csv}"
OUT_MD="${OUT_MD:-${REPO_ROOT}/benchmarks/hard_cases_report.md}"
OUT_JSON="${OUT_JSON:-${REPO_ROOT}/benchmarks/hard_cases_report.json}"

cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ "${RUN_BENCHMARK:-0}" = "1" ]; then
    "${PYTHON_BIN}" -m visual_memory.benchmarks.full_benchmark \
        --dataset "${REPO_ROOT}/benchmarks/dataset.csv" \
        --images "${REPO_ROOT}/benchmarks/images" \
        --no-fp-holdout-tests \
        --no-fp-expanded-tests \
        "$@"
else
    if [ "$#" -gt 0 ]; then
        echo "Arguments are only passed to full_benchmark when RUN_BENCHMARK=1." >&2
        exit 2
    fi
fi

"${PYTHON_BIN}" scripts/report_hard_cases.py \
    --manifest "${MANIFEST}" \
    --results-json "${RESULTS_JSON}" \
    --traces-csv "${TRACES_CSV}" \
    --output-md "${OUT_MD}" \
    --output-json "${OUT_JSON}"
