#!/usr/bin/env bash
set -euo pipefail
# MA17-v1: complete MA11 versus MA11 plus same-stream native column AllGather.
# Both arms retain MA8-v2, MA9 and original MA11 forward/backward. Preview unless --execute.
ma17_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma17_runtime=/usr/local/cuda-12.4
if [[ -f "$ma17_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma17_repo/outputs/h100/container-recovery/env.sh"
    ma17_runtime=$CUDA_HOME
fi
ma17_python=${MA17_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma17_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma17_repo"
exec "$ma17_python" tools/h100/phase6/detach_ma17.py \
    --cuda-home "$ma17_runtime" --build-cuda-home "$ma17_compiler" "$@"
