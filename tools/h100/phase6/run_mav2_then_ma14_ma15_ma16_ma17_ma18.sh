#!/usr/bin/env bash
set -euo pipefail
# Compatibility filename: one detached supervisor owns MAV2, then MA14 -> MA20.
mav2_sequence_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$mav2_sequence_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$mav2_sequence_repo/outputs/h100/container-recovery/env.sh"
fi
mav2_sequence_python=${H100_SEQUENCE_PYTHON:-${MAV2_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}}
cd -- "$mav2_sequence_repo"
exec "$mav2_sequence_python" tools/h100/phase6/detach_mav2_then_ma14_ma18.py "$@"
