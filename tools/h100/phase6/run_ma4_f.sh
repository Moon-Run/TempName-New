#!/usr/bin/env bash
set -euo pipefail
ma4_f_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma4_f_runtime=/usr/local/cuda-12.4
if [[ -f "$ma4_f_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma4_f_repo/outputs/h100/container-recovery/env.sh"
    ma4_f_runtime=$CUDA_HOME
fi
ma4_f_python=${MA4_F_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma4_f_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma4_f_repo"
exec "$ma4_f_python" tools/h100/phase6/detach_ma4_f.py \
    --cuda-home "$ma4_f_runtime" --build-cuda-home "$ma4_f_compiler" "$@"
