#!/usr/bin/env bash
set -euo pipefail
# Explicit comparison entrypoint; identical gates and protocol to run_op3.sh.
op3_compare_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$op3_compare_here/run_op3.sh" "$@"
