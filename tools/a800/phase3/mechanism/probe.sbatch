#!/usr/bin/env bash
#SBATCH --job-name=flux-mechanism-probe
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:03:00
#SBATCH --output=logs/a800/phase3/mechanism/probe-%j.log
set -uo pipefail
out="$SLURM_SUBMIT_DIR/logs/a800/phase3/mechanism/$SLURM_JOB_ID"
mkdir -p "$out"
hostname > "$out/hostname.txt"
nvidia-smi > "$out/nvidia-smi.txt"
/data/apps/cuda/12.8/bin/ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct --csv --log-file "$out/counters.csv" "$SLURM_SUBMIT_DIR/outputs/a800/phase3/counter_probe" > "$out/probe.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$out/ncu-exit-code.txt"
exit "$rc"
