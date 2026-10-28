#!/usr/bin/env bash
#SBATCH --job-name=flux-a800-three-way
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/a800/phase3/three-way-%j.log
set -euo pipefail
export PHASE3_CONFIG=three_way.json
export PHASE3_BUILD_DIR="$SLURM_SUBMIT_DIR/outputs/a800/phase3/three-way-build"
export PHASE3_IPC_CHECK=0
exec bash "$SLURM_SUBMIT_DIR/tools/a800/phase3/extension.sbatch" 4
