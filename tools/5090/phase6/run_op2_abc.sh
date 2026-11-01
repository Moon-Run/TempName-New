#!/usr/bin/env bash
set -euo pipefail
# Independent OP2 A/B/C with MA89 and MA11 optimizations. Preview unless --execute is supplied.
op2_abc_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$op2_abc_repo/outputs/5090/env-setup/env.sh" ]]; then
    source "$op2_abc_repo/outputs/5090/env-setup/env.sh"
fi
op2_abc_python=${RTX5090_PYTHON:-${RTX5090_SETUP_PYTHON:-python3}}
op2_abc_cuda=${RTX5090_CUDA_HOME:-${RTX5090_SETUP_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}}
op2_abc_build_cuda=${RTX5090_BUILD_CUDA_HOME:-$op2_abc_cuda}
op2_abc_megatron=${RTX5090_MEGATRON:-${RTX5090_SETUP_MEGATRON:-$op2_abc_repo/../Megatron-LM}}
cd -- "$op2_abc_repo"
export PYTHONDONTWRITEBYTECODE=1
exec "$op2_abc_python" tools/5090/phase6/op2_abc.py \
    --cuda-home "$op2_abc_cuda" --build-cuda-home "$op2_abc_build_cuda" \
    --megatron "$op2_abc_megatron" "$@"
