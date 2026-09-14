#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Full comparison: 3 models x 4 devices x 4 lookbacks = 48 experiments.
for model in Informer iTransformer PatchTST; do
    bash "$SCRIPT_DIR/run_all.sh" "$@" --model "$model"
done
