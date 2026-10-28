#!/usr/bin/env bash
#SBATCH --job-name=flux-e2e-distributed
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with distributed/submit.py --submit}"
: "${E2E_DIST_OUT:?}"
cd "$E2E_DIST_ROOT"
mkdir -p "$E2E_DIST_OUT/scripts"
cp tools/a800/e2e/distributed/*.py tools/a800/e2e/distributed/*.sbatch "$E2E_DIST_OUT/scripts/"
export CUDA_HOME="${MEGATRON_CUDA_HOME:-/data/apps/cuda/12.4}"
export PATH="$(dirname "$E2E_DIST_PYTHON"):$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
# Respect explicitly selected network interfaces; do not force IB off or invent NIC names.
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${E2E_MASTER_PORT:-$((20000 + SLURM_JOB_ID % 30000))}"
scontrol show job "$SLURM_JOB_ID" > "$E2E_DIST_OUT/slurm-job.txt"
finish() { rc=$?; printf '%s\n' "$rc" > "$E2E_DIST_OUT/exit-code.txt"; }
trap finish EXIT
"$E2E_DIST_PYTHON" "$E2E_DIST_OUT/scripts/launch.py"
