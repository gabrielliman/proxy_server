#!/bin/bash
ulimit -n 524288

export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export MODEL_PATH="/scratch/hpc4ai/models/llama/Llama-3.1-8B-Instruct"

source /opt/conda/etc/profile.d/conda.sh && conda activate proxy_server
set -x

wait_for_ready() {
  local PORT=$1
  echo "[INFO] Waiting for model on port $PORT to become ready..."
  until curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; do
    sleep 5
  done
  echo "[INFO] Model on port $PORT is ready!"
}

export LOG_DIR="./var/logs"
mkdir -p "$LOG_DIR"

# 1. Sobe o servidor UMA vez, com max-num-seqs alto o bastante
#    pra cobrir o maior batch size que você vai testar
CUDA_VISIBLE_DEVICES=0 python -u -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "8000" \
        --max-num-seqs "1024" \
        --dtype bfloat16 \
        --max-model-len 10000 \
        --gpu-memory-utilization 0.9 \
        --served-model-name "$MODEL" \
        > >(tee "$LOG_DIR/saida_VLLM_8000.txt") 2>&1 &

SERVER_PID=$!
wait_for_ready 8000

# 2. Varredura de batch sizes (== max-concurrency)
BATCH_SIZES=(1 2 4 8 16 32 48 64 96 128 192 256 384 512 768 1024)

RESULTS_CSV="$LOG_DIR/throughput_sweep.csv"
echo "batch_size,output_tok_s,total_tok_s,request_throughput_s" > "$RESULTS_CSV"

for BS in "${BATCH_SIZES[@]}"; do

  RESULT_JSON="result_bs${BS}.json"

  vllm bench serve \
        --backend vllm \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len 2 \
        --random-output-len 1900 \
        --ignore-eos \
        --num-prompts "$BS" \
        --request-rate inf \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,75,90,99 \
        --save-result \
        --result-dir "$LOG_DIR" \
        --result-filename "$RESULT_JSON" \
        > "${LOG_DIR}/saida_benchmark_bs${BS}.txt" 2>&1

  # 3. Extrai as métricas de throughput do JSON salvo
  python3 - "$LOG_DIR/$RESULT_JSON" "$BS" "$RESULTS_CSV" <<'EOF'
import json, sys

result_path, bs, csv_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(result_path) as f:
    d = json.load(f)

out_tok_s = d.get("output_throughput", "NA")
total_tok_s = d.get("total_token_throughput", "NA")
req_s = d.get("request_throughput", "NA")

with open(csv_path, "a") as f:
    f.write(f"{bs},{out_tok_s},{total_tok_s},{req_s}\n")
EOF

done

# 4. Encerra o servidor
echo "[INFO] Varredura completa. Encerrando o servidor vLLM (PID: $SERVER_PID)..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "[INFO] Servidor encerrado."

cat "$RESULTS_CSV"