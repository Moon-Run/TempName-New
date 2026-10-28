#!/usr/bin/env bash
set -euo pipefail
# MA12-v1: MA89 versus MA89 plus tile-scoped publication.
# Both arms retain exact MA8-v2 caching and MA9; only the candidate changes native publication. Preview unless --execute.
ma12_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma12_runtime=/usr/local/cuda-12.4
if [[ -f "$ma12_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma12_repo/outputs/h100/container-recovery/env.sh"
    ma12_runtime=$CUDA_HOME
fi
ma12_python=${MA12_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma12_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma12_repo"
exec "$ma12_python" tools/h100/phase6/detach_ma12.py \
    --cuda-home "$ma12_runtime" --build-cuda-home "$ma12_compiler" "$@"
