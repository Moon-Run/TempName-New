#!/usr/bin/env bash
set -euo pipefail
# MA19-v1: complete MA11 versus MA11 plus fixed-descriptor master-to-model parameter publication.
# Both arms retain MA8-v2, MA9 and original MA11 forward/backward. Preview unless --execute.
ma19_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma19_runtime=/usr/local/cuda-12.4
if [[ -f "$ma19_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma19_repo/outputs/h100/container-recovery/env.sh"
    ma19_runtime=$CUDA_HOME
fi
ma19_python=${MA19_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma19_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma19_repo"
exec "$ma19_python" tools/h100/phase6/detach_ma19.py \
    --cuda-home "$ma19_runtime" --build-cuda-home "$ma19_compiler" "$@"
