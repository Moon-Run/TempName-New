#!/usr/bin/env bash
set -euo pipefail
# MA6-v3: fixed R1 geometry; independently compare packed mask and CTA layout.
# All three arms retain the original F4 backward. Preview unless --execute.
ma6_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma6_runtime=/usr/local/cuda-12.4
if [[ -f "$ma6_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma6_repo/outputs/h100/container-recovery/env.sh"
    ma6_runtime=$CUDA_HOME
fi
ma6_python=${MA6_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma6_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma6_repo"
exec "$ma6_python" tools/h100/phase6/detach_ma6.py \
    --cuda-home "$ma6_runtime" --build-cuda-home "$ma6_compiler" "$@"
