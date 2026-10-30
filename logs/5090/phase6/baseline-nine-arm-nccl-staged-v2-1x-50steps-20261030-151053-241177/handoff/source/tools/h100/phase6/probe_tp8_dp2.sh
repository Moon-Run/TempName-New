#!/usr/bin/env bash
set -euo pipefail
# Run once inside each training container. No training or collectives are launched.
probe_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
probe_python=${TP8_DP2_PROBE_PYTHON:-python3}
exec "$probe_python" "$probe_repo/tools/h100/phase6/probe_tp8_dp2.py" "$@"
