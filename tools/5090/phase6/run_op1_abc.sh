#!/usr/bin/env bash
set -euo pipefail
# Independent OP1 A/B/C and Flux improved quantization comparison. Preview unless --execute is supplied.
op1_abc_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$op1_abc_repo/outputs/5090/env-setup/env.sh" ]]; then
    source "$op1_abc_repo/outputs/5090/env-setup/env.sh"
fi
op1_abc_python=${RTX5090_PYTHON:-${RTX5090_SETUP_PYTHON:-python3}}
op1_abc_cuda=${RTX5090_CUDA_HOME:-${RTX5090_SETUP_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}}
op1_abc_build_cuda=${RTX5090_BUILD_CUDA_HOME:-$op1_abc_cuda}
op1_abc_megatron=${RTX5090_MEGATRON:-${RTX5090_SETUP_MEGATRON:-$op1_abc_repo/../Megatron-LM}}
cd -- "$op1_abc_repo"
export PYTHONDONTWRITEBYTECODE=1
exec "$op1_abc_python" tools/5090/phase6/op1_abc.py \
    --cuda-home "$op1_abc_cuda" --build-cuda-home "$op1_abc_build_cuda" \
    --megatron "$op1_abc_megatron" "$@"
