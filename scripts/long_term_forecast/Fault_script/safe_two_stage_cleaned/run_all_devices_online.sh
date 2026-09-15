#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/dataset/fault}"
CLASSIFICATION_ROOT="${CLASSIFICATION_ROOT:-$PROJECT_ROOT/outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/safe_all_devices_evaluation}"
PROFILE_END="${PROFILE_END:?Set PROFILE_END, for example '2014-01-01 00:00:00'}"
TEST_END="${TEST_END:?Set TEST_END, for example '2014-03-11 11:34:27'}"
DEVICE="${DEVICE:-cuda:0}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -u run_safe_online_evaluation.py \
    --data-root "$DATA_ROOT" --classification-root "$CLASSIFICATION_ROOT" \
    --lengths 3 6 12 24 --profile-end "$PROFILE_END" --test-end "$TEST_END" \
    --selection-policy same_device_then_shared --output-root "$OUTPUT_ROOT" \
    --plan-only "$@"