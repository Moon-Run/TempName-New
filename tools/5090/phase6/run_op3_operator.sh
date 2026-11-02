#!/usr/bin/env bash
set -euo pipefail
# Same preflight/build/native gates; stop after combined operator measurements.
op3_operator_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$op3_operator_here/run_op3.sh" --operator-only "$@"
