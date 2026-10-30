#!/usr/bin/env bash
set -euo pipefail
# Complete MA11 versus fixed native C++ autograd linear plans; preview by default.
mav1_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
mav1_runtime=/usr/local/cuda-12.4
if [[ -f "$mav1_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$mav1_repo/outputs/h100/container-recovery/env.sh"
    mav1_runtime=$CUDA_HOME
fi
if [[ -n "${MAV1_PYTHON:-}" ]]; then
    mav1_python=$MAV1_PYTHON
elif [[ -n "${H100_RECOVERY_PYTHON:-}" ]]; then
    mav1_python=$H100_RECOVERY_PYTHON
elif [[ -x /workspace/conda-envs/flux-h100-ablation3-20261017/bin/python ]]; then
    mav1_python=/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python
else
    mav1_python=$(command -v python3)
fi
mav1_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$mav1_repo"
exec "$mav1_python" tools/h100/phase6/detach_mav1.py \
    --cuda-home "$mav1_runtime" --build-cuda-home "$mav1_compiler" "$@"
