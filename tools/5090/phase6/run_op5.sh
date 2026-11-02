#!/usr/bin/env bash
set -euo pipefail
# OP5 v2: packet All-to-All/P2P-LL and AllGather/CE probes, then complete training.
op5_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$op5_repo/outputs/5090/env-setup/env.sh" ]]; then
    source "$op5_repo/outputs/5090/env-setup/env.sh"
fi
op5_python=${RTX5090_PYTHON:-${RTX5090_SETUP_PYTHON:-python3}}
op5_cuda=${RTX5090_CUDA_HOME:-${RTX5090_SETUP_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}}
op5_build_cuda=${RTX5090_BUILD_CUDA_HOME:-$op5_cuda}
op5_megatron=${RTX5090_MEGATRON:-${RTX5090_SETUP_MEGATRON:-$op5_repo/../Megatron-LM}}
cd -- "$op5_repo"
export PYTHONDONTWRITEBYTECODE=1
exec "$op5_python" tools/5090/phase6/op5.py \
    --cuda-home "$op5_cuda" --build-cuda-home "$op5_build_cuda" \
    --megatron "$op5_megatron" "$@"
