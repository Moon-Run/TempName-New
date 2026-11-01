#!/usr/bin/env bash
set -euo pipefail
# OP3 native testing (--transport-smoke-only) or complete two-arm experiment.
# Preview only unless --execute is supplied; all results use a fresh output directory.
op3_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$op3_repo/outputs/5090/env-setup/env.sh" ]]; then
    source "$op3_repo/outputs/5090/env-setup/env.sh"
fi
op3_python=${RTX5090_PYTHON:-${RTX5090_SETUP_PYTHON:-python3}}
op3_cuda=${RTX5090_CUDA_HOME:-${RTX5090_SETUP_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}}
op3_build_cuda=${RTX5090_BUILD_CUDA_HOME:-$op3_cuda}
op3_megatron=${RTX5090_MEGATRON:-${RTX5090_SETUP_MEGATRON:-$op3_repo/../Megatron-LM}}
cd -- "$op3_repo"
export PYTHONDONTWRITEBYTECODE=1
exec "$op3_python" tools/5090/phase6/op3.py \
    --cuda-home "$op3_cuda" --build-cuda-home "$op3_build_cuda" \
    --megatron "$op3_megatron" "$@"
