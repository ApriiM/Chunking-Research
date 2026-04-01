DATASET=031_dense_x_retrieval_poquad_validation_merged
apptainer exec \
    --nv \
    --writable-tmpfs \
    --pwd /app \
    --bind "$(pwd)/results/success/${DATASET}/session_20260326T001202Z:/app/${DATASET}" \
    --bind "$(pwd)/../pirb_data:/app/out" \
    chunkowanie.sif \
    bash -c "python3 run_annotate_and_convert.py \
        --input-path ${DATASET} \
        --output-root out"