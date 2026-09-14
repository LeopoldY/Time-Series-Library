#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for device_id in 27 58 69 83; do
    bash "$SCRIPT_DIR/run_device.sh" "$device_id" "$@"
done
