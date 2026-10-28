#!/usr/bin/env bash
set -euo pipefail
# MA8-v2: unchanged CR3/F4 versus retained column-parallel activations.
# F4 column path is untouched; candidate has no added per-call diagnostics. Preview unless --execute.
ma8_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma8_runtime=/usr/local/cuda-12.4
if [[ -f "$ma8_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma8_repo/outputs/h100/container-recovery/env.sh"
    ma8_runtime=$CUDA_HOME
fi
ma8_python=${MA8_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma8_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma8_repo"
exec "$ma8_python" tools/h100/phase6/detach_ma8.py \
    --cuda-home "$ma8_runtime" --build-cuda-home "$ma8_compiler" "$@"
