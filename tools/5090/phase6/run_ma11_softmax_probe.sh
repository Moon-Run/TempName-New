#!/usr/bin/env bash
set -euo pipefail
rtx5090_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -f "$rtx5090_repo/outputs/5090/env-setup/env.sh" ]]; then
    source "$rtx5090_repo/outputs/5090/env-setup/env.sh"
fi
rtx5090_python=${RTX5090_PYTHON:-${RTX5090_SETUP_PYTHON:-python3}}
rtx5090_cuda=${RTX5090_CUDA_HOME:-${RTX5090_SETUP_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}}
cd -- "$rtx5090_repo"
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$rtx5090_cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$rtx5090_python" tools/5090/phase6/ma11_softmax_probe.py --cuda-home "$rtx5090_cuda" "$@"
