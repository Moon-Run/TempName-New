#!/usr/bin/env bash
set -euo pipefail
# MA16-v1: complete MA11 versus MA11 plus local-first column AllGather/GEMM overlap.
# Both arms retain MA8-v2, MA9 and original MA11 forward/backward. Preview unless --execute.
ma16_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma16_runtime=/usr/local/cuda-12.4
if [[ -f "$ma16_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma16_repo/outputs/h100/container-recovery/env.sh"
    ma16_runtime=$CUDA_HOME
fi
ma16_python=${MA16_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma16_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma16_repo"
exec "$ma16_python" tools/h100/phase6/detach_ma16.py \
    --cuda-home "$ma16_runtime" --build-cuda-home "$ma16_compiler" "$@"
