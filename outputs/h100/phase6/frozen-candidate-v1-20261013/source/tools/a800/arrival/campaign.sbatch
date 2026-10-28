#!/usr/bin/env bash
#SBATCH --job-name=tile-v2-tp4
#SBATCH --partition=gpu_a800
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
set -euo pipefail
RUN_ROOT=${1:?absolute experiment directory required}
REPO=${2:?absolute repository directory required}
cd "$REPO"
export TZ=Asia/Shanghai
date -Is > "$RUN_ROOT/started-at.txt"
scontrol show job "$SLURM_JOB_ID" > "$RUN_ROOT/slurm-job.txt"
finish() { rc=$?; printf '%s\n' "$rc" > "$RUN_ROOT/job-exit-code.txt"; exit "$rc"; }
trap finish EXIT
# sbatch can dispatch before the submitting process finishes saving its receipt.
for attempt in {1..30}; do
  [[ -f "$RUN_ROOT/submission.json" ]] && break
  sleep 1
done
test -f "$RUN_ROOT/submission.json"
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus=4 --kill-on-bad-exit=1 \
  bash -c 'hostname; nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv; nvidia-smi topo -m' \
  > "$RUN_ROOT/gpus.log" 2>&1
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus=4 --kill-on-bad-exit=1 \
  "$REPO/outputs/a800/venv/bin/python" \
  "$RUN_ROOT/source/tools/a800/arrival/campaign.py" "$RUN_ROOT" "$REPO"
# Normal return (or failure via set -e) releases this short allocation.
