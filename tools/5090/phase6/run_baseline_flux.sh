#!/usr/bin/env bash
set -euo pipefail
# Independent four-arm Flux quantization comparison. Preview unless --execute is supplied.
flux_four_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$flux_four_repo/outputs/5090/env-setup/env.sh" ]]; then
    source "$flux_four_repo/outputs/5090/env-setup/env.sh"
fi
flux_four_python=${RTX5090_PYTHON:-${RTX5090_SETUP_PYTHON:-python3}}
flux_four_cuda=${RTX5090_CUDA_HOME:-${RTX5090_SETUP_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}}
flux_four_build_cuda=${RTX5090_BUILD_CUDA_HOME:-$flux_four_cuda}
flux_four_megatron=${RTX5090_MEGATRON:-${RTX5090_SETUP_MEGATRON:-$flux_four_repo/../Megatron-LM}}
cd -- "$flux_four_repo"
export PYTHONDONTWRITEBYTECODE=1
exec "$flux_four_python" tools/5090/phase6/flux_baseline.py \
    --cuda-home "$flux_four_cuda" --build-cuda-home "$flux_four_build_cuda" \
    --megatron "$flux_four_megatron" "$@"
