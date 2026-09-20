#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# Pass --train to execute. Defaults compare 3 backbones x 5 devices x 4 lengths x 3 seeds.
python -u run_safe_fault_types.py --models iTransformer Informer PatchTST "$@"
