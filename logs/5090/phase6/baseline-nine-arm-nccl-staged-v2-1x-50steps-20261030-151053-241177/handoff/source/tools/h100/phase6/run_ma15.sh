#!/usr/bin/env bash
set -euo pipefail
# MA15-v1: complete MA11 versus MA11 plus fixed-shape attention CUDA Graph replay.
# Both arms retain MA8-v2, MA9 and original MA11 forward/backward. Preview unless --execute.
ma15_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma15_runtime=/usr/local/cuda-12.4
if [[ -f "$ma15_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma15_repo/outputs/h100/container-recovery/env.sh"
    ma15_runtime=$CUDA_HOME
fi
ma15_python=${MA15_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma15_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma15_repo"
exec "$ma15_python" tools/h100/phase6/detach_ma15.py \
    --cuda-home "$ma15_runtime" --build-cuda-home "$ma15_compiler" "$@"
