#!/usr/bin/env bash
set -euo pipefail
# Independent three-arm OP1-A experiment. Preview unless --execute is supplied.
op1_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$op1_repo/outputs/5090/env-setup/env.sh" ]]; then
    source "$op1_repo/outputs/5090/env-setup/env.sh"
fi
op1_python=${RTX5090_PYTHON:-${RTX5090_SETUP_PYTHON:-python3}}
op1_cuda=${RTX5090_CUDA_HOME:-${RTX5090_SETUP_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}}
op1_build_cuda=${RTX5090_BUILD_CUDA_HOME:-$op1_cuda}
op1_megatron=${RTX5090_MEGATRON:-${RTX5090_SETUP_MEGATRON:-$op1_repo/../Megatron-LM}}
cd -- "$op1_repo"
export PYTHONDONTWRITEBYTECODE=1
exec "$op1_python" tools/5090/phase6/op1_a.py \
    --cuda-home "$op1_cuda" --build-cuda-home "$op1_build_cuda" \
    --megatron "$op1_megatron" "$@"
