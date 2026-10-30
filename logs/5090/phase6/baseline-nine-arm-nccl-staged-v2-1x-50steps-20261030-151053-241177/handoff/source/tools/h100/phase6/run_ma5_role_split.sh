#!/usr/bin/env bash
# MA5-R uses the same recovered H100 environment as run_basic_quant.sh.
set -euo pipefail
ma5r_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma5r_runtime=/usr/local/cuda-12.4
if [[ -f "$ma5r_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma5r_repo/outputs/h100/container-recovery/env.sh"
    ma5r_runtime=$CUDA_HOME
fi
ma5r_python=${MA5R_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma5r_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma5r_repo"
exec "$ma5r_python" tools/h100/phase6/detach_ma5r.py \
    --cuda-home "$ma5r_runtime" --build-cuda-home "$ma5r_compiler" "$@"
