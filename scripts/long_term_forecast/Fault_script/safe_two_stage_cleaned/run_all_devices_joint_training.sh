#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hp/miniconda3/envs/tslib/bin/python}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/dataset/fault_raw/processed_all_cleaned/four_levels}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/safe_joint_routed_training}"
DEVICE="${DEVICE:-cuda:0}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -u run_safe_joint_routed_training.py \
    --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" \
    --lengths 3 6 12 24 --epochs 100 --patience 5 \
    --batch-size 32 --learning-rate 1e-4 --joint-loss-weight 1.0 \
    --regression-loss mse --device "$DEVICE" --train "$@"