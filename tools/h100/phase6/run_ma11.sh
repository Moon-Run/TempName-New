#!/usr/bin/env bash
set -euo pipefail
# MA11-v1: MA89 versus MA89 plus causal attention softmax.
# Both arms retain exact MA8-v2 caching and MA9; only the candidate adds MA11. Preview unless --execute.
ma11_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma11_runtime=/usr/local/cuda-12.4
if [[ -f "$ma11_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma11_repo/outputs/h100/container-recovery/env.sh"
    ma11_runtime=$CUDA_HOME
fi
ma11_python=${MA11_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma11_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma11_repo"
exec "$ma11_python" tools/h100/phase6/detach_ma11.py \
    --cuda-home "$ma11_runtime" --build-cuda-home "$ma11_compiler" "$@"
