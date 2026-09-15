#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/dataset/fault_selected_cleaned}"
BACKBONE_ROOT="${BACKBONE_ROOT:-$PROJECT_ROOT/outputs/safe_two_stage_cleaned}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/safe_stage2_classification}"

cd "$PROJECT_ROOT"

for device_model in \
    '27 iTransformer' '58 Informer' '69 iTransformer' '83 PatchTST'; do
    read -r device_id model <<< "$device_model"
    for seq_len in 3 6 12 24; do
        matches=("$BACKBONE_ROOT/device${device_id}_${model}_L${seq_len}"/*/stage1_regression/best_backbone.pt)
        if [[ ${#matches[@]} -ne 1 || ! -f ${matches[0]} ]]; then
            printf 'Expected exactly one backbone for device=%s model=%s seq_len=%s, found %s\n' \
                "$device_id" "$model" "$seq_len" "${#matches[@]}" >&2
            exit 1
        fi
        checkpoint="${matches[0]}"
        config_path="$(dirname "$(dirname "$checkpoint")")/config.json"
        if [[ ! -f "$config_path" || ! -f "$DATA_ROOT/设备${device_id}_cleaned.csv" ]]; then
            printf 'Missing config or data for device=%s model=%s seq_len=%s\n' \
                "$device_id" "$model" "$seq_len" >&2
            exit 1
        fi

        mapfile -t config_args < <("$PYTHON_BIN" - "$config_path" <<'PY'
import json
import sys

config = json.loads(open(sys.argv[1], encoding='utf-8').read())
for key in ('d_model', 'n_heads', 'd_ff', 'batch_size', 'seed',
            'regression_loss', 'head_hidden', 'head_dropout'):
    value = config.get(key)
    if value is not None:
        print(f'--{key.replace("_", "-")}\n{value}')
PY
        )

        echo "[head] device=$device_id model=$model seq_len=$seq_len checkpoint=$checkpoint"
        "$PYTHON_BIN" -u run_safe_two_stage.py \
            --device-id "$device_id" --model "$model" --seq-len "$seq_len" \
            --stage head --backbone-checkpoint "$checkpoint" \
            --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" \
            "${config_args[@]}" "$@"
    done
done