#!/bin/bash
#SBATCH --job-name=chunking-with-llm
#SBATCH --partition=lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:30:00
#SBATCH --array=0-1
#SBATCH --output=logs2/slurm_log_%A_%a.out


mapfile -t CONFIGS < tasks_lumberchunker.txt
CURRENT_CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

VLLM_APPTAINER="vllm.sif"
CHUNKING_APPTAINER="chunkowanie_final.sif"

CHOSEN_MODEL="/lustre/pd03/hpc-tomasznaskret-1743501581/chunking-research/chunking/models/gpt-oss-20b"
DISPLAYED_MODEL_NAME="openai/gpt-oss-20b"
HF_HUB_CACHE="/lustre/pd03/hpc-tomasznaskret-1743501581/chunking-research/chunking/models"

export VLLM_PORT=$(( 8000 + SLURM_ARRAY_TASK_ID * 100 ))
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

echo "--- Start zadania ---"
echo "ID Podzadania (Array): $SLURM_ARRAY_TASK_ID"
echo "Używana konfiguracja: $CURRENT_CONFIG"

mkdir -p $TMPDIR/.cache/vllm
mkdir -p $TMPDIR/.cache/triton
mkdir -p $TMPDIR/.cache/tiktoken
mkdir -p $TMPDIR/.cache/xdg

CUSTOM_ENV="$TMPDIR/.env_override"
echo "LUMBERCHUNKER_API_MODEL=openai/gpt-oss-20b" >> $CUSTOM_ENV
echo "LUMBERCHUNKER_API_BASE_URL=http://localhost:${VLLM_PORT}/v1" > $CUSTOM_ENV
echo "OPENAI_API_KEY=ZSA12345" >> $CUSTOM_ENV


echo "--> Podnoszenie serwera vLLM z kontenera $VLLM_APPTAINER (Port: $VLLM_PORT)..."
apptainer exec \
    --nv \
    --ipc \
    --bind /dev/shm:/dev/shm \
    --bind "$HF_HUB_CACHE:$HF_HUB_CACHE" \
    --bind "$TMPDIR/.cache:/workspace" \
    $VLLM_APPTAINER \
    bash -c "
        export VLLM_USE_V1=0
        export NCCL_SHM_DISABLE=1
        export VLLM_WORKER_MULTIPROC_METHOD=spawn
        
        export TRITON_HOME=/workspace/triton
        export VLLM_CACHE_ROOT=/workspace/vllm
        export XDG_CACHE_HOME=/workspace/xdg
        export HOME=/workspace
        export TIKTOKEN_CACHE_DIR=/workspace/tiktoken

        export TMPDIR=/tmp
        export TMP=/tmp
        export TEMP=/tmp
        
        python3 -m vllm.entrypoints.openai.api_server \
            --model $CHOSEN_MODEL \
            --served-model-name $DISPLAYED_MODEL_NAME \
            --max-model-len 24000 \
            --gpu-memory-utilization 0.90 \
            --trust-remote-code \
            --port $VLLM_PORT \
            --disable-custom-all-reduce \
            --enforce-eager
    " > vllm_server_${SLURM_ARRAY_TASK_ID}.log 2>&1 &

VLLM_PID=$!

echo "--> Czekam na załadowanie modelu do VRAM..."
until curl -s http://localhost:${VLLM_PORT}/v1/models | grep -q "id"; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "CRITICAL: Serwer vLLM niespodziewanie zakończył pracę."
        tail -n 30 vllm_server_${SLURM_ARRAY_TASK_ID}.log
        exit 1
    fi
    sleep 10
done
echo "--> Serwer vLLM jest gotowy!"


echo "--> Uruchamianie eksperymentu w kontenerze $CHUNKING_APPTAINER..."
apptainer exec \
    --nv \
    --writable-tmpfs \
    --pwd /app \
    --bind "$(pwd)/results:/app/results" \
    --bind "$CUSTOM_ENV:/app/.env" \
    $CHUNKING_APPTAINER \
    bash -c "
        echo \"--> Instalacja kompatybilnej wersji numpy...\"
        pip install \"numpy<2\" -t /tmp/python_packages > /dev/null 2>&1
        export PYTHONPATH=\"/tmp/python_packages:\$PYTHONPATH\"
        
        python3 run_experiment.py --config /app/configs/experiments/full_pipeline_exps/${CURRENT_CONFIG}
    "

echo "--> Eksperyment zakończony. Zatrzymywanie vLLM..."
kill $VLLM_PID
echo "All done."