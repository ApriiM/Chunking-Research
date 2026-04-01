#!/bin/sh
#SBATCH --job-name=chunking
#SBATCH --partition=lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:30:00
#SBATCH --array=0-1
#SBATCH --output=logs2/slurm_log_%A_%a.out  # %A to ID głównego zadania, %a to ID podzadania

mapfile -t CONFIGS < tasks.txt

CURRENT_CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "--- Start zadania ---"
echo "ID Podzadania: $SLURM_ARRAY_TASK_ID"
echo "Używana konfiguracja: $CURRENT_CONFIG"
apptainer exec \
    --nv \
    --writable-tmpfs \
    --pwd /app \
    --bind "$(pwd)/results:/app/results" \
    chunkowanie_final.sif \
    bash -c "python3 run_experiment.py --config /app/configs/experiments/full_pipeline_exps/${CURRENT_CONFIG}"