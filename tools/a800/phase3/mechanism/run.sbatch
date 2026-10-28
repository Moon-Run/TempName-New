#!/usr/bin/env bash
#SBATCH --job-name=flux-mechanism-tp4
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/a800/phase3/mechanism/run-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR"
export RESULT_DIR="$ROOT/logs/a800/phase3/mechanism/$SLURM_JOB_ID"
mkdir -p "$RESULT_DIR/scripts"
cp "$ROOT"/tools/a800/phase3/mechanism/*.py "$ROOT"/tools/a800/phase3/mechanism/*.h "$ROOT"/tools/a800/phase3/mechanism/*.cu "$ROOT"/tools/a800/phase3/mechanism/*.json "$ROOT"/tools/a800/phase3/mechanism/*.sbatch "$RESULT_DIR/scripts/"
finish(){ rc=$?; printf '%s\n' "$rc" > "$RESULT_DIR/exit-code.txt"; nvidia-smi --query-gpu=index,uuid,temperature.gpu,clocks.sm,clocks.mem --format=csv > "$RESULT_DIR/gpus-after.csv" || true; exit "$rc"; }
trap finish EXIT
export CUDA_HOME=/data/apps/cuda/12.8
export PATH="$ROOT/outputs/a800/venv/bin:$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=NVL FLUX_FORCE_NVLINK=1
hostname > "$RESULT_DIR/hostname.txt"
nvidia-smi topo -m > "$RESULT_DIR/topology.txt"
grep -q NV8 "$RESULT_DIR/topology.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem --format=csv > "$RESULT_DIR/gpus.csv"
git rev-parse HEAD > "$RESULT_DIR/commit.txt"
git diff > "$RESULT_DIR/worktree.patch"
scontrol show job "$SLURM_JOB_ID" > "$RESULT_DIR/slurm-job.txt"
python "$RESULT_DIR/scripts/launch.py" 2>&1 | tee "$RESULT_DIR/benchmark.log"
