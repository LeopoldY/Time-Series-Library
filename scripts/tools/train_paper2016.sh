#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-dataset/fault_paper2016/four_levels}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/safe_paper2016}"
"$PYTHON_BIN" scripts/tools/preflight_paper2016.py --data-root "$DATA_ROOT"
for fold in {0..9}; do
  "$PYTHON_BIN" run_safe_joint_routed_training.py --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT/fold$fold" --cv-fold "$fold" --train "$@"
done
