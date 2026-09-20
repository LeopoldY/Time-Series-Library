#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# Pass --train to execute. Defaults: 5 original device routes x L=6/12 x one seed = 10 two-stage jobs.
python -u run_safe_fault_types.py "$@"
