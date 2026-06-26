BASE_URL="http://localhost:8081"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export CUSTOM_API_BASE="$BASE_URL"
export CUSTOM_MODEL="$MODEL_NAME"
python LanguageAgentTreeSearch/hotpot/run.py \
    --algorithm lats \
    --task_start_index 900 \
    --task_end_index 901 \
    --iterations 1 \
    --n_generate_sample 100 \
    --n_evaluate_sample 1 \
    --output-json teste1.json \
    --burstiness 1 \
    --program-rate 1 \
    --is_baseline_run 0

# curl -s "http://localhost:8081/processes_summary"