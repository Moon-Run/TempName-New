#!/usr/bin/env bash
# MA5 uses the same recovered H100 environment as run_basic_quant.sh.
set -euo pipefail
ma5_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma5_runtime=/usr/local/cuda-12.4
if [[ -f "$ma5_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma5_repo/outputs/h100/container-recovery/env.sh"
    ma5_runtime=$CUDA_HOME
fi
ma5_python=${MA5_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma5_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma5_repo"
exec "$ma5_python" tools/h100/phase6/detach_ma5.py \
    --cuda-home "$ma5_runtime" --build-cuda-home "$ma5_compiler" "$@"
