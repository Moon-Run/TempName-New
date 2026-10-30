#!/usr/bin/env bash
set -euo pipefail
# MAV2: Flux basic H8, selective D8/H8 and forced D8; all remote groups remain quantized.
# All three arms share GemmOnly, 136-byte packets, NCCL and ordered BF16 reduction. Preview unless --execute.
# --allow-quality-failure continues timing while retaining failed training-quality verdicts.
mav2_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
mav2_runtime=/usr/local/cuda-12.4
if [[ -f "$mav2_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$mav2_repo/outputs/h100/container-recovery/env.sh"
    mav2_runtime=$CUDA_HOME
fi
mav2_python=${MAV2_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
mav2_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$mav2_repo"
exec "$mav2_python" tools/h100/phase6/detach_mav2.py \
    --cuda-home "$mav2_runtime" --build-cuda-home "$mav2_compiler" "$@"
