#!/usr/bin/env bash
#SBATCH --job-name=flux-arrival-numeric
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/a800/arrival/run-%j.log
set -euo pipefail
export ARRIVAL_STAGE=numerical-diagnostic
ROOT="$SLURM_SUBMIT_DIR"
export RESULT_DIR="$ROOT/logs/a800/arrival/$SLURM_JOB_ID"
mkdir -p "$RESULT_DIR/scripts"
cp "$ROOT"/tools/a800/arrival/*.* "$RESULT_DIR/scripts/"
finish(){ rc=$?; printf '%s\n' "$rc" > "$RESULT_DIR/exit-code.txt"; nvidia-smi --query-gpu=index,uuid,temperature.gpu,clocks.sm,clocks.mem --format=csv > "$RESULT_DIR/gpus-after.csv" || true; exit "$rc"; }
trap finish EXIT
export CUDA_HOME=/data/apps/cuda/12.8
export PATH="$ROOT/outputs/a800/venv/bin:$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=NVL FLUX_FORCE_NVLINK=1
hostname > "$RESULT_DIR/hostname.txt"
nvidia-smi topo -m > "$RESULT_DIR/topology.txt"
grep -q NV8 "$RESULT_DIR/topology.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem --format=csv > "$RESULT_DIR/gpus.csv"
scontrol show job "$SLURM_JOB_ID" > "$RESULT_DIR/slurm-job.txt"
git -C "$ROOT" rev-parse HEAD > "$RESULT_DIR/commit.txt"
python "$RESULT_DIR/scripts/diagnose_numeric.py" --launch
