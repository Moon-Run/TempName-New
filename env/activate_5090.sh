#!/usr/bin/env bash
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    echo 'Use: source env/activate_5090.sh' >&2
    exit 1
fi
rtx5090_env_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
rtx5090_env_generated="$rtx5090_env_repo/outputs/5090/env-setup/env.sh"
if [[ ! -f "$rtx5090_env_generated" ]]; then
    echo 'Run bash env/setup_5090.sh successfully first.' >&2
    unset rtx5090_env_repo rtx5090_env_generated
    return 1
fi
source "$rtx5090_env_generated"
source "$RTX5090_SETUP_PREFIX/bin/activate"
export RTX5090_PYTHON="$RTX5090_SETUP_PYTHON"
export RTX5090_CUDA_HOME="$RTX5090_SETUP_CUDA_HOME"
export RTX5090_BUILD_CUDA_HOME="$RTX5090_SETUP_CUDA_HOME"
export RTX5090_MEGATRON="$RTX5090_SETUP_MEGATRON"
export CUDA_HOME="$RTX5090_SETUP_CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$RTX5090_SETUP_MEGATRON"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 NCCL_NVLS_ENABLE=0
unset PYTHONHOME PYTHONOPTIMIZE rtx5090_env_repo rtx5090_env_generated
