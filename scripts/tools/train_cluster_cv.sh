#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-dataset/fault_paper2016/four_levels}"
RUN_DIR="${RUN_DIR:-outputs/safe_cluster_cv/$(date +%Y%m%d_%H%M%S)}"
exec "$PYTHON_BIN" -u run_safe_cluster_cv.py --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR" --train "$@"
