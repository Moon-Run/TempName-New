#!/usr/bin/env bash
set -euo pipefail
# MA14-v2: complete MA11 versus MA11 plus epoch publication with native pair ownership.
# Both arms retain MA8-v2, MA9 and original MA11 forward/backward. Preview unless --execute.
ma14_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma14_runtime=/usr/local/cuda-12.4
if [[ -f "$ma14_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma14_repo/outputs/h100/container-recovery/env.sh"
    ma14_runtime=$CUDA_HOME
fi
ma14_python=${MA14_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma14_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma14_repo"
exec "$ma14_python" tools/h100/phase6/detach_ma14.py \
    --cuda-home "$ma14_runtime" --build-cuda-home "$ma14_compiler" "$@"
