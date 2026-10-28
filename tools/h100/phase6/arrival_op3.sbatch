#!/usr/bin/env bash
#SBATCH --job-name=arrival-op3-validate
#SBATCH --partition=gpu_h100
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
set -euo pipefail

# Explicit arguments avoid inheriting old gxn75 calibration identities.
# sbatch this-script /absolute/repository /absolute/compiled-A-O /new/result/path [micro|fullstep]
OP3_REPO=${1:?absolute repository required}
OP3_BUILD=${2:?compiled build containing A and O required}
OP3_OUT=${3:?new output path required}
OP3_SUITE=${4:-micro}
OP3_PYTHON=${OP3_PYTHON:-/data/run01/scyb672/conda_envs/flux-megatron-a800/bin/python}
OP3_CUDA=${OP3_CUDA:-/data/apps/cuda/12.8}
cd "$OP3_REPO"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCCL_NVLS_ENABLE=0

# The worker records and validates actual H100 model/SM count, UUIDs and P2P.
# A partition name alone is not considered hardware validation.
case "$OP3_SUITE" in
  micro)
    "$OP3_PYTHON" tools/h100/phase6/arrival_op3_bench.py prepare "$OP3_OUT" --build "$OP3_BUILD"
    srun --nodes=1 --ntasks=1 --gpus=8 --cpus-per-task=16 --kill-on-bad-exit=1 \
      "$OP3_PYTHON" tools/h100/phase6/arrival_op3_bench.py run "$OP3_OUT" \
      --python "$OP3_PYTHON" --cuda-home "$OP3_CUDA" --execute
    srun --nodes=1 --ntasks=1 --gpus=8 --cpus-per-task=16 --kill-on-bad-exit=1 \
      "$OP3_PYTHON" tools/h100/phase6/arrival_op3_bench.py run "$OP3_OUT" \
      --python "$OP3_PYTHON" --cuda-home "$OP3_CUDA" --profile --execute
    ;;
  fullstep)
    "$OP3_PYTHON" tools/h100/phase6/arrival_op3_fullstep.py prepare "$OP3_OUT" --build "$OP3_BUILD"
    srun --nodes=1 --ntasks=1 --gpus=8 --cpus-per-task=16 --kill-on-bad-exit=1 \
      "$OP3_PYTHON" tools/h100/phase6/arrival_op3_fullstep.py run "$OP3_OUT" \
      --python "$OP3_PYTHON" --cuda-home "$OP3_CUDA" --execute
    ;;
  *)
    echo "Expected micro or fullstep, got: $OP3_SUITE" >&2
    exit 2
    ;;
esac
