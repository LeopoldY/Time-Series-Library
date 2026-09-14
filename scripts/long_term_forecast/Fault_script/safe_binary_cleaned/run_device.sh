#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
DEVICE_ID="${1:?Usage: bash run_device.sh DEVICE_ID [--train] [runner options]}"
shift
case "$DEVICE_ID" in 27|58|69|83) ;; *) echo 'Allowed devices: 27 58 69 83' >&2; exit 2;; esac
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/dataset/fault_selected_cleaned}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/safe_binary_cleaned}"
cd "$PROJECT_ROOT"
for seq_len in 3 6 12 24; do
    "$PYTHON_BIN" -u "$PROJECT_ROOT/run_safe_binary.py" \
        --device-id "$DEVICE_ID" --seq-len "$seq_len" \
        --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" "$@"
done
