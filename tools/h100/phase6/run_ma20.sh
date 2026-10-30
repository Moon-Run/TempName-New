#!/usr/bin/env bash
set -euo pipefail
# MA20: complete MA11 control, joint codec-first order/mask bank, fresh heldout selection.
ma20_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma20_runtime=/usr/local/cuda-12.4
if [[ -f "$ma20_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma20_repo/outputs/h100/container-recovery/env.sh"
    ma20_runtime=$CUDA_HOME
fi
ma20_python=${MA20_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma20_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma20_repo"
exec "$ma20_python" tools/h100/phase6/detach_ma20.py \
    --cuda-home "$ma20_runtime" --build-cuda-home "$ma20_compiler" "$@"
