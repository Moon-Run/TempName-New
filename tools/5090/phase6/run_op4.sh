#!/usr/bin/env bash
set -euo pipefail
# OP4 candidate only: operator checks and measurements, then complete training.
op4_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$op4_repo/outputs/5090/env-setup/env.sh" ]]; then
    source "$op4_repo/outputs/5090/env-setup/env.sh"
fi
op4_python=${RTX5090_PYTHON:-${RTX5090_SETUP_PYTHON:-python3}}
op4_cuda=${RTX5090_CUDA_HOME:-${RTX5090_SETUP_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}}
op4_build_cuda=${RTX5090_BUILD_CUDA_HOME:-$op4_cuda}
op4_megatron=${RTX5090_MEGATRON:-${RTX5090_SETUP_MEGATRON:-$op4_repo/../Megatron-LM}}
cd -- "$op4_repo"
export PYTHONDONTWRITEBYTECODE=1
exec "$op4_python" tools/5090/phase6/op4.py \
    --cuda-home "$op4_cuda" --build-cuda-home "$op4_build_cuda" \
    --megatron "$op4_megatron" "$@"
