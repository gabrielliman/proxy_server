#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/global/abacus
BASE_URL="http://localhost:8080"
DATASET="/scratch/global/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
MODEL="Qwen/Qwen3-4B"
LIMIT=10

METRICS_URL="http://localhost:8106/metrics"
RESET_URL="http://localhost:8106/reset_prefix_cache"

OUTPUT_LOG="prefix_hit_rate_test.csv"
OUTPUTS_DIR="outputs_benchmark"
FINAL_CSV="benchmark_dataset_full.csv"

echo "mode,max_chat_len,run,metrics" > "$OUTPUT_LOG"


MODES=("completion" "chat")
CHAT_LENS=(1 3 5 10 25)
REPEATS=3

for MODE in "${MODES[@]}"; do
  for CHAT_LEN in "${CHAT_LENS[@]}"; do
    for RUN in $(seq 1 $REPEATS); do

      echo "Executando: mode=$MODE | max_chat_len=$CHAT_LEN | run=$RUN"

      python benchmark_stateful.py \
        --base-url "$BASE_URL" \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --limit "$LIMIT" \
        --mode "$MODE" \
        --max_chat_len "$CHAT_LEN" \
        --output-json "./${OUTPUTS_DIR}/output_${MODE}_${CHAT_LEN}_run${RUN}.json"

      METRICS=$(curl -s "$METRICS_URL" \
        | grep -E '^vllm:prefix_cache_(queries|hits)_total' \
        | tr '\n' ' ')

      # Salva no arquivo
      echo "$MODE,$CHAT_LEN,$RUN,\"$METRICS\"" >> "$OUTPUT_LOG"

      # Reset do prefix cache
      curl -s -X POST "$RESET_URL" > /dev/null

      echo "Reset do prefix cache realizado"
      echo "--------------------------------------"

    done
  done
done

echo "Todos os testes foram concluídos."

python fix.py \
  --input-csv "$OUTPUT_LOG" \
  --outputs-dir "$OUTPUTS_DIR" \
  --output-csv "$FINAL_CSV"