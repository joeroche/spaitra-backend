#!/bin/bash
set -e

REPO_ROOT="${1:-/opt/spaitra/backend-copy}"
cd "$REPO_ROOT"
source "$REPO_ROOT/venv-core/bin/activate"
python -m visual_memory.utils.memory_monitor --cleanup --max-age 2 --log-only
