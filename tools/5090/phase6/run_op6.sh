#!/usr/bin/env bash
set -euo pipefail
op6_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$op6_repo/outputs/5090/env-setup/env.sh" ]]; then
    source "$op6_repo/outputs/5090/env-setup/env.sh"
fi
op6_python=${RTX5090_PYTHON:-${RTX5090_SETUP_PYTHON:-python3}}
op6_cuda=${RTX5090_CUDA_HOME:-${RTX5090_SETUP_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}}
op6_build_cuda=${RTX5090_BUILD_CUDA_HOME:-$op6_cuda}
op6_megatron=${RTX5090_MEGATRON:-${RTX5090_SETUP_MEGATRON:-$op6_repo/../Megatron-LM}}
cd -- "$op6_repo"
export PYTHONDONTWRITEBYTECODE=1
exec "$op6_python" tools/5090/phase6/op6.py \
    --cuda-home "$op6_cuda" --build-cuda-home "$op6_build_cuda" \
    --megatron "$op6_megatron" "$@"
