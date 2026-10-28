#!/usr/bin/env bash
# One basic/CUDA comparison. Reuse the recovered container environment if present.
set -euo pipefail
basic_quant_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
basic_quant_runtime=/usr/local/cuda-12.4
if [[ -f "$basic_quant_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$basic_quant_repo/outputs/h100/container-recovery/env.sh"
    basic_quant_runtime=$CUDA_HOME
fi
basic_quant_python=${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}
basic_quant_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$basic_quant_repo"
exec "$basic_quant_python" tools/h100/phase6/detach_basic_quant.py \
    --cuda-home "$basic_quant_runtime" --build-cuda-home "$basic_quant_compiler" "$@"
