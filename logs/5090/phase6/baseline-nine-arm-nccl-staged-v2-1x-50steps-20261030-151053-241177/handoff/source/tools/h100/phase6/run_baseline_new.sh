#!/usr/bin/env bash
set -euo pipefail
# Single-node five-arm baseline. No GPU work unless --execute is supplied.
baseline_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$baseline_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$baseline_repo/outputs/h100/container-recovery/env.sh"
fi
baseline_python=${BASELINE_NEW_PYTHON:-${H100_RECOVERY_PYTHON:-$baseline_repo/../conda-envs/tempname-h100/bin/python}}
baseline_runtime=${BASELINE_NEW_CUDA_HOME:-/usr/local/cuda-12.4}
baseline_compiler=${BASELINE_NEW_BUILD_CUDA:-${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}}
cd -- "$baseline_repo"
exec "$baseline_python" tools/h100/phase6/baseline_new.py \
    --cuda-home "$baseline_runtime" --build-cuda-home "$baseline_compiler" "$@"
