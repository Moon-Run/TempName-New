#!/usr/bin/env bash
#SBATCH --job-name=flux-megatron-e2e
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/a800/e2e/comparison-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit to the ordinary gpu_a800 partition.}"
ROOT="$SLURM_SUBMIT_DIR"
export RESULT_DIR="$ROOT/logs/a800/e2e/comparison/$SLURM_JOB_ID"
export E2E_STAGE="${E2E_STAGE:-validate}"
mkdir -p "$RESULT_DIR/scripts"
cp "$ROOT"/tools/a800/e2e/*.py "$ROOT"/tools/a800/e2e/*.json "$ROOT"/tools/a800/e2e/*.sbatch "$RESULT_DIR/scripts/"
finish() {
 rc=$?
 printf '%s\n' "$rc" > "$RESULT_DIR/exit-code.txt"
 nvidia-smi --query-gpu=index,uuid,temperature.gpu,clocks.sm,clocks.mem --format=csv > "$RESULT_DIR/gpus-after.csv" || true
 exit "$rc"
}
trap finish EXIT
export CUDA_HOME=/data/apps/cuda/12.4
export PATH="$(cd "$ROOT/../.." && pwd -P)/conda_envs/flux-megatron-a800/bin:$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2
export CUDA_DEVICE_MAX_CONNECTIONS=1 FLUX_FORCE_NVLINK=1
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=NVL NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
hostname > "$RESULT_DIR/hostname.txt"
nvidia-smi topo -m > "$RESULT_DIR/topology.txt"
grep -q NV8 "$RESULT_DIR/topology.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem --format=csv > "$RESULT_DIR/gpus.csv"
scontrol show job "$SLURM_JOB_ID" > "$RESULT_DIR/slurm-job.txt"
git -C "$ROOT" rev-parse HEAD > "$RESULT_DIR/flux-commit.txt"
git -C "$ROOT/../Megatron-LM" rev-parse HEAD > "$RESULT_DIR/megatron-commit.txt"
git -C "$ROOT/../Megatron-LM" diff > "$RESULT_DIR/megatron-worktree.patch"
python -m pip freeze > "$RESULT_DIR/python-packages.txt"
python "$RESULT_DIR/scripts/launch.py" 2>&1 | tee "$RESULT_DIR/launch.log"
