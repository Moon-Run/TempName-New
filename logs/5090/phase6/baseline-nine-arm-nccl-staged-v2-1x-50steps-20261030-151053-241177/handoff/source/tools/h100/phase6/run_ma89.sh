#!/usr/bin/env bash
set -euo pipefail
# MA89-v1: MA8-v2 F4-cache versus F4-cache plus exact optimizer zero-count fusion.
# Both arms retain MA8-v2 caching; only the candidate adds MA9. Preview unless --execute.
ma89_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma89_runtime=/usr/local/cuda-12.4
if [[ -f "$ma89_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma89_repo/outputs/h100/container-recovery/env.sh"
    ma89_runtime=$CUDA_HOME
fi
ma89_python=${MA89_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma89_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma89_repo"
exec "$ma89_python" tools/h100/phase6/detach_ma89.py \
    --cuda-home "$ma89_runtime" --build-cuda-home "$ma89_compiler" "$@"
