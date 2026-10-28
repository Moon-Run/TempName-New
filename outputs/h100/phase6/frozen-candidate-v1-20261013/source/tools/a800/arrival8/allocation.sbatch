#!/usr/bin/env bash
#SBATCH --job-name=tile-v2-tp8-hold
#SBATCH --partition=gpu_a800
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
set -euo pipefail

# One allocation owns all internal steps. A failed/completed campaign does not
# end this outer batch process. RELEASE is an explicit request to release it.
RUN_ROOT=${1:?absolute campaign directory required}
REPO=${2:?absolute repository directory required}
LOG_ROOT="$RUN_ROOT/allocation-$SLURM_JOB_ID"
mkdir -p "$LOG_ROOT"
cd "$REPO"
export TZ=Asia/Shanghai
date -Is > "$LOG_ROOT/allocated-at.txt"
scontrol show job "$SLURM_JOB_ID" > "$LOG_ROOT/slurm-job.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$RUN_ROOT/allocation-job-id.txt"
state() { printf '%s\n' "$1" > "$RUN_ROOT/allocation-state.txt"; }
trap 'state allocation_ending' EXIT
state validating_allocation
set +e
srun --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=8 --kill-on-bad-exit=1 \
  bash -c 'hostname; nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv; nvidia-smi topo -m' \
  > "$LOG_ROOT/gpus.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$LOG_ROOT/resource-check-exit-code.txt"
state waiting_for_campaign
while [[ ! -f "$RUN_ROOT/RELEASE" ]]; do
  if [[ -f "$RUN_ROOT/campaign-ready.sh" && ! -f "$LOG_ROOT/campaign-exit-code.txt" ]]; then
    state running_campaign
    set +e
    srun --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=8 --kill-on-bad-exit=1 \
      bash "$RUN_ROOT/campaign-ready.sh" > "$LOG_ROOT/campaign.log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$LOG_ROOT/campaign-exit-code.txt"
    if [[ "$rc" == 0 ]]; then state campaign_completed_holding; else state campaign_failed_holding; fi
  fi
  sleep 30
done
state release_requested
