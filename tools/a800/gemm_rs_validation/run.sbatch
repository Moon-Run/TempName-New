#!/usr/bin/env bash
#SBATCH --job-name=flux-bf16-gemm-rs
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/data/run01/scyb672/hjr/TempName/outputs/a800/gemm_rs_validation/logs/run-%j.log
set -euo pipefail
ROOT=/data/run01/scyb672/hjr/TempName
OUT="$ROOT/outputs/a800/gemm_rs_validation"
SRC="$OUT/source"
export RESULT_DIR="$OUT/results/$SLURM_JOB_ID"
mkdir -p "$RESULT_DIR"
export CUDA_HOME=/data/apps/cuda/12.8
export PATH="$ROOT/outputs/a800/venv/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$SRC/python:$SRC"
export LD_LIBRARY_PATH="$SRC/build/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG_FILE="$RESULT_DIR/nccl-%h-%p.log"
# Keep the verified single-node NVLink path fixed and avoid network transports.
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=NVL FLUX_FORCE_NVLINK=1
cd "$SRC"
hostname > "$RESULT_DIR/hostname.txt"
nvidia-smi topo -m > "$RESULT_DIR/topology.txt"
grep -q NV8 "$RESULT_DIR/topology.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,clocks.sm,clocks.mem,power.limit --format=csv > "$RESULT_DIR/gpus.csv"
"$OUT/nccl_gpu_check" 2>&1 | tee "$RESULT_DIR/nccl-project.log"
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$OUT/scripts/benchmark.py" 2>&1 | tee "$RESULT_DIR/benchmark.log"
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu --format=csv > "$RESULT_DIR/gpus-after.csv"
python "$OUT/scripts/summarize.py" "$RESULT_DIR"
echo "RESULT_DIR=$RESULT_DIR"
