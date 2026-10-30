#!/usr/bin/env bash
set -euo pipefail
# MA18-v1: complete MA11 versus MA11 plus IPC DMA column AllGather.
# Both arms retain MA8-v2, MA9 and original MA11 forward/backward. Preview unless --execute.
ma18_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma18_runtime=/usr/local/cuda-12.4
if [[ -f "$ma18_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma18_repo/outputs/h100/container-recovery/env.sh"
    ma18_runtime=$CUDA_HOME
fi
ma18_python=${MA18_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma18_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma18_repo"
exec "$ma18_python" tools/h100/phase6/detach_ma18.py \
    --cuda-home "$ma18_runtime" --build-cuda-home "$ma18_compiler" "$@"
