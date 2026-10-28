#!/usr/bin/env bash
#SBATCH --job-name=flux-a800-phase2
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/a800/phase2/run-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR"
BASE="$ROOT/outputs/a800/gemm_rs_validation/source"
BUILD="$ROOT/outputs/a800/phase2/build"
export RESULT_DIR="$ROOT/logs/a800/phase2/$SLURM_JOB_ID"
mkdir -p "$RESULT_DIR/scripts"
cp "$ROOT"/tools/a800/phase2/* "$RESULT_DIR/scripts/"
cp "$BUILD/instrumentation.patch" "$BUILD/commands.json" "$RESULT_DIR/"
export CUDA_HOME=/data/apps/cuda/12.8
export PATH="$ROOT/outputs/a800/venv/bin:$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=NVL FLUX_FORCE_NVLINK=1
export PHASE2_LIBRARY="$BUILD/libflux_cuda.so"
hostname > "$RESULT_DIR/hostname.txt"
nvidia-smi topo -m > "$RESULT_DIR/topology.txt"
grep -q NV8 "$RESULT_DIR/topology.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,clocks.sm,clocks.mem --format=csv > "$RESULT_DIR/gpus.csv"
sha256sum "$BASE/build/lib/libflux_cuda.so" "$BUILD/libflux_cuda.so" > "$RESULT_DIR/build-sha256.txt"
for kind in original_before instrumented original_after; do
    export PHASE2_KIND="$kind"
    if [[ "$kind" == instrumented ]]; then
        export PYTHONPATH="$BUILD/python:$BASE/python"
        export LD_LIBRARY_PATH="$BUILD:$BASE/build/lib:$CUDA_HOME/lib64"
    else
        export PYTHONPATH="$BASE/python"
        export LD_LIBRARY_PATH="$BASE/build/lib:$CUDA_HOME/lib64"
    fi
    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        "$ROOT/tools/a800/phase2/run.py" 2>&1 | tee "$RESULT_DIR/$kind.log"
done
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks.mem --format=csv > "$RESULT_DIR/gpus-after.csv"
python "$ROOT/tools/a800/phase2/summarize.py" "$RESULT_DIR" > "$RESULT_DIR/analysis.log"
echo "RESULT_DIR=$RESULT_DIR"
