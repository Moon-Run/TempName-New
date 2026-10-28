#!/usr/bin/env bash
set -euo pipefail
# MA7-v1: unchanged CR3/F4 versus deferred wgrad auxiliary stream.
# Native forward stays F4 for both arms. Preview unless --execute.
ma7_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ma7_runtime=/usr/local/cuda-12.4
if [[ -f "$ma7_repo/outputs/h100/container-recovery/env.sh" ]]; then
    source "$ma7_repo/outputs/h100/container-recovery/env.sh"
    ma7_runtime=$CUDA_HOME
fi
ma7_python=${MA7_PYTHON:-${H100_RECOVERY_PYTHON:-/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python}}
ma7_compiler=${H100_RECOVERY_BUILD_CUDA:-/usr/local/cuda-12.8}
cd -- "$ma7_repo"
exec "$ma7_python" tools/h100/phase6/detach_ma7.py \
    --cuda-home "$ma7_runtime" --build-cuda-home "$ma7_compiler" "$@"
