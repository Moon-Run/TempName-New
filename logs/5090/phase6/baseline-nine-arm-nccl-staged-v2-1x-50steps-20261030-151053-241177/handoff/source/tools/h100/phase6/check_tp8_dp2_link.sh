#!/usr/bin/env bash
set -euo pipefail
# Preview by default. --execute launches 16 distributed communication-test workers.
link_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
link_python=${TP8_DP2_PROBE_PYTHON:-python3}
exec "$link_python" "$link_repo/tools/h100/phase6/check_tp8_dp2_link.py" "$@"
