#!/usr/bin/env bash
#SBATCH --job-name=flux-a800-phase1
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/a800/phase1/run-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR"
SRC="$ROOT/outputs/a800/gemm_rs_validation/source"
export RESULT_DIR="$ROOT/logs/a800/phase1/$SLURM_JOB_ID"
mkdir -p "$RESULT_DIR"
export CUDA_HOME=/data/apps/cuda/12.8
export PATH="$ROOT/outputs/a800/venv/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$SRC/python:$SRC"
export LD_LIBRARY_PATH="$SRC/build/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG_FILE="$RESULT_DIR/nccl-%h-%p.log"
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=NVL FLUX_FORCE_NVLINK=1
hostname > "$RESULT_DIR/hostname.txt"
nvidia-smi topo -m > "$RESULT_DIR/topology.txt"
grep -q NV8 "$RESULT_DIR/topology.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,clocks.sm,clocks.mem,power.limit --format=csv > "$RESULT_DIR/gpus.csv"
git -C "$ROOT" rev-parse HEAD > "$RESULT_DIR/commit.txt"
sha256sum "$SRC/build/lib/libflux_cuda.so" "$SRC/build/lib/libflux_cuda_ths_op.so" "$SRC"/python/flux_ths_pybind*.so > "$RESULT_DIR/build-sha256.txt"
cp "$ROOT/tools/a800/phase1.py" "$ROOT/tools/a800/phase1.sbatch" "$RESULT_DIR/"
cd "$SRC"
python -m torch.distributed.run --standalone --nproc_per_node=2 \
    "$ROOT/tools/a800/phase1.py" 2>&1 | tee "$RESULT_DIR/benchmark.log"
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu --format=csv > "$RESULT_DIR/gpus-after.csv"
python "$ROOT/tools/a800/summarize_phase1.py" "$RESULT_DIR" > "$RESULT_DIR/analysis.log"
echo "RESULT_DIR=$RESULT_DIR"
