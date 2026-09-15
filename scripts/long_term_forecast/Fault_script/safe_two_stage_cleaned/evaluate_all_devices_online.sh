#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
PLAN="${1:?Usage: evaluate_all_devices_online.sh PLAN_DIR/plan.json [--allow-partial]}"
shift

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -u run_safe_online_evaluation.py \
    --plan "$PLAN" --evaluate-only --device "$DEVICE" "$@"