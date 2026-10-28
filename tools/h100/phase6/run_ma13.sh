#!/usr/bin/env bash
set -euo pipefail
# MA13-v1: complete MA11 versus MA11 plus resident-exponential softmax forward.
# Both arms retain MA8-v2, MA9 and original MA11 backward. Preview unless --execute.
ma13_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma13_runtime=/usr/local/cuda-12.4
if [[ -f "$ma13_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma13_repo/outputs/h100/container-recovery/env.sh"
    ma13_runtime=$CUDA_HOME
fi
ma13_python=${MA13_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma13_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma13_repo"
exec "$ma13_python" tools/h100/phase6/detach_ma13.py \
    --cuda-home "$ma13_runtime" --build-cuda-home "$ma13_compiler" "$@"
