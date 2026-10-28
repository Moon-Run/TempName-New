#!/usr/bin/env bash
#SBATCH --job-name=a800-tp4-hold-2d
#SBATCH --partition=gpu_a800
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
set -euo pipefail

# The outer batch allocation outlives every internal test, including failures.
# A campaign-ready.sh is optional; an empty control directory only holds GPUs.
ALLOCATION_ROOT=${1:?absolute allocation control directory required}
REPO_ROOT=${2:?absolute repository directory required}
[[ "$ALLOCATION_ROOT" == /* && "$REPO_ROOT" == /* ]]
LOG_ROOT="$ALLOCATION_ROOT/allocation-$SLURM_JOB_ID"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"
export TZ=Asia/Shanghai
state() { printf '%s\n' "$1" > "$ALLOCATION_ROOT/allocation-state.txt"; }
finish() {
  local rc=$?
  printf '%s\n' "$rc" > "$LOG_ROOT/batch-exit-code.txt"
  if [[ ! -f "$ALLOCATION_ROOT/RELEASE" ]]; then state allocation_ending; fi
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
date -Is > "$LOG_ROOT/allocated-at.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$ALLOCATION_ROOT/allocation-job-id.txt"
scontrol show job "$SLURM_JOB_ID" > "$LOG_ROOT/slurm-job.txt" || true
state validating_allocation
rc=0
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus=4 --kill-on-bad-exit=1 \
  bash -c 'hostname; nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv; nvidia-smi topo -m' \
  > "$LOG_ROOT/gpus.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$LOG_ROOT/resource-check-exit-code.txt"
if [[ "$rc" == 0 ]]; then state waiting_for_campaign; else state resource_check_failed_holding; fi
while [[ ! -f "$ALLOCATION_ROOT/RELEASE" ]]; do
  if [[ -f "$ALLOCATION_ROOT/campaign-ready.sh" && ! -f "$LOG_ROOT/campaign-exit-code.txt" ]]; then
    state running_campaign
    rc=0
    srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus=4 --kill-on-bad-exit=1 \
      bash "$ALLOCATION_ROOT/campaign-ready.sh" > "$LOG_ROOT/campaign.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" > "$LOG_ROOT/campaign-exit-code.txt"
    if [[ "$rc" == 0 ]]; then state campaign_completed_holding; else state campaign_failed_holding; fi
  fi
  date -Is > "$LOG_ROOT/heartbeat.txt"
  sleep 30
done
state release_requested
