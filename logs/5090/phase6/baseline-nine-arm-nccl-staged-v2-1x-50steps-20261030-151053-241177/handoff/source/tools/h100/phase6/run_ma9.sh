#!/usr/bin/env bash
set -euo pipefail
# MA9-v1: unchanged CR3/F4 versus fused optimizer zero-count reductions.
# Native forward stays F4 for both arms. Preview unless --execute.
ma9_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma9_runtime=/usr/local/cuda-12.4
if [[ -f "$ma9_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma9_repo/outputs/h100/container-recovery/env.sh"
    ma9_runtime=$CUDA_HOME
fi
ma9_python=${MA9_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma9_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma9_repo"
exec "$ma9_python" tools/h100/phase6/detach_ma9.py \
    --cuda-home "$ma9_runtime" --build-cuda-home "$ma9_compiler" "$@"
