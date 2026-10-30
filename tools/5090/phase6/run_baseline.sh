#!/usr/bin/env bash
set -euo pipefail
# Nine-arm RTX5090 TP8 campaign. No GPU queries or writes without --execute.
rtx5090_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$rtx5090_repo/outputs/5090/env-setup/env.sh" ]]; then
    source "$rtx5090_repo/outputs/5090/env-setup/env.sh"
fi
rtx5090_python=${RTX5090_PYTHON:-${RTX5090_SETUP_PYTHON:-python3}}
rtx5090_cuda=${RTX5090_CUDA_HOME:-${RTX5090_SETUP_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}}
rtx5090_build_cuda=${RTX5090_BUILD_CUDA_HOME:-$rtx5090_cuda}
rtx5090_megatron=${RTX5090_MEGATRON:-${RTX5090_SETUP_MEGATRON:-$rtx5090_repo/../Megatron-LM}}
cd -- "$rtx5090_repo"
export PYTHONDONTWRITEBYTECODE=1
exec "$rtx5090_python" tools/5090/phase6/baseline.py \
    --cuda-home "$rtx5090_cuda" --build-cuda-home "$rtx5090_build_cuda" \
    --megatron "$rtx5090_megatron" "$@"
