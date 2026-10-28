#!/usr/bin/env bash
#SBATCH --job-name=flux-a800-phase3
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/a800/phase3/run-%j.log
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR"
BASE="$ROOT/outputs/a800/gemm_rs_validation/source"
BUILD="$ROOT/outputs/a800/phase3/build"
export RESULT_DIR="$ROOT/logs/a800/phase3/$SLURM_JOB_ID"
mkdir -p "$RESULT_DIR/scripts"
cp "$ROOT"/tools/a800/phase3/*.py "$ROOT"/tools/a800/phase3/*.cu "$ROOT"/tools/a800/phase3/*.sbatch "$RESULT_DIR/scripts/"
cp "$BUILD/manifest.json" "$BUILD/commands.json" "$RESULT_DIR/"
export CUDA_HOME=/data/apps/cuda/12.8
export PATH="$ROOT/outputs/a800/venv/bin:$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=NVL FLUX_FORCE_NVLINK=1
hostname > "$RESULT_DIR/hostname.txt"
nvidia-smi topo -m > "$RESULT_DIR/topology.txt"
grep -q NV8 "$RESULT_DIR/topology.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,clocks.sm,clocks.mem,power.limit --format=csv > "$RESULT_DIR/gpus.csv"
for policy in original remote_first interleaved; do
 variant="$policy"
 if [[ "$policy" == original ]]; then variant=original_matched; fi
 cp "$BUILD/$variant/swizzle.patch" "$RESULT_DIR/$policy.patch"
 "$BUILD/$variant/check_mapping" > "$RESULT_DIR/mapping-$policy.csv" 2> "$RESULT_DIR/mapping-$policy.log"
done
# Every permutation appears once; each policy occupies each position twice.
orders=("original remote_first interleaved" "remote_first interleaved original" "interleaved original remote_first" "original interleaved remote_first" "interleaved remote_first original" "remote_first original interleaved")
window=0
for block in "${!orders[@]}"; do
 for policy in ${orders[$block]}; do
  variant="$policy"
  if [[ "$policy" == original ]]; then variant=original_matched; fi
  export PHASE3_POLICY="$policy" PHASE3_BLOCK="$block" PHASE3_WINDOW="$window"
  export PHASE3_LIBRARY="$BUILD/$variant/libflux_cuda.so"
  export PYTHONPATH="$BUILD/$variant/python:$BASE/python"
  export LD_LIBRARY_PATH="$BUILD/$variant:$BASE/build/lib:$CUDA_HOME/lib64"
  export PHASE3_DIAGNOSTIC=0
  if [[ "$block" == 0 ]]; then export PHASE3_DIAGNOSTIC=1; fi
  python -m torch.distributed.run --standalone --nproc_per_node=2 \
   "$ROOT/tools/a800/phase3/run.py" 2>&1 | tee "$RESULT_DIR/window-$window-$policy.log"
  window=$((window+1))
 done
done
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks.mem --format=csv > "$RESULT_DIR/gpus-after.csv"
python "$ROOT/tools/a800/phase3/summarize.py" "$RESULT_DIR" > "$RESULT_DIR/analysis.log"
echo "RESULT_DIR=$RESULT_DIR"
