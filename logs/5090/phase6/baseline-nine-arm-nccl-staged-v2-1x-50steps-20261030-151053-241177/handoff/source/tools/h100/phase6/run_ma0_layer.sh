#!/usr/bin/env bash
set -euo pipefail
# Single-node gxn75, TP8 DP1. Defaults to a read-only preview.
layer_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$layer_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$layer_repo/outputs/h100/container-recovery/env.sh"
fi
layer_python=${MA0_LAYER_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
layer_cuda=${MA0_LAYER_CUDA_HOME:-${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}}
cd -- "$layer_repo"
exec "$layer_python" tools/h100/phase6/ma0_layer.py --cuda-home "$layer_cuda" "$@"
