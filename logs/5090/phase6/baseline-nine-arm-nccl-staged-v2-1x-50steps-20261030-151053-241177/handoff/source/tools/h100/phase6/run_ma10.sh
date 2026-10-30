#!/usr/bin/env bash
set -euo pipefail
# MA10-v1: unchanged CR3/F4 versus bounded DDP gradient accumulation batches.
# Native forward stays F4 for both arms. Preview unless --execute.
ma10_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma10_runtime=/usr/local/cuda-12.4
if [[ -f "$ma10_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma10_repo/outputs/h100/container-recovery/env.sh"
    ma10_runtime=$CUDA_HOME
fi
ma10_python=${MA10_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma10_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma10_repo"
exec "$ma10_python" tools/h100/phase6/detach_ma10.py \
    --cuda-home "$ma10_runtime" --build-cuda-home "$ma10_compiler" "$@"
