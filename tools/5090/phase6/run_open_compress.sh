#!/usr/bin/env bash
set -euo pipefail
open_compress_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$open_compress_repo/outputs/5090/env-setup/env.sh" ]]; then
    source "$open_compress_repo/outputs/5090/env-setup/env.sh"
fi
open_compress_python=${RTX5090_PYTHON:-${RTX5090_SETUP_PYTHON:-python3}}
open_compress_cuda=${RTX5090_CUDA_HOME:-${RTX5090_SETUP_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}}
open_compress_build_cuda=${RTX5090_BUILD_CUDA_HOME:-$open_compress_cuda}
open_compress_megatron=${RTX5090_MEGATRON:-${RTX5090_SETUP_MEGATRON:-$open_compress_repo/../Megatron-LM}}
cd -- "$open_compress_repo"
export PYTHONDONTWRITEBYTECODE=1
exec "$open_compress_python" tools/5090/phase6/open_compress.py \
    --cuda-home "$open_compress_cuda" --build-cuda-home "$open_compress_build_cuda" \
    --megatron "$open_compress_megatron" "$@"
