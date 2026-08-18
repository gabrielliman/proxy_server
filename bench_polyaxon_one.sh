#!/bin/bash
ulimit -n 524288

# Define environment variables
export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export MODEL_PATH="/scratch/hpc4ai/models/llama/Llama-3.1-8B-Instruct"
export NUM_PROMPTS=100000
export REQUEST_RATE="inf"

source /opt/conda/etc/profile.d/conda.sh && conda activate proxy_server
set -x

# Define the health-check function
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
# 1. Start the server in the background (Notice the trailing '&' and removed '\')
CUDA_VISIBLE_DEVICES=0 python -u -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "8000" \
        --max-num-seqs "1024" \
        --dtype bfloat16 \
        --max-model-len $NUM_PROMPTS \
        --gpu-memory-utilization 0.9 \
        --served-model-name "meta-llama/Llama-3.1-8B-Instruct" \
        > >(tee "$LOG_DIR/saida_VLLM_8000.txt") 2>&1 &

# 2. Capture the background server's Process ID so we can kill it later
SERVER_PID=$!

# 3. Call the wait function to pause the script until the server responds
wait_for_ready 8000

# 4. Run the benchmark
echo "[INFO] Starting the benchmark..."
vllm bench serve \
        --backend vllm \
        --model $MODEL \
        --num-prompts $NUM_PROMPTS \
        --dataset-name random \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,75,90,99 \
        --request-rate $REQUEST_RATE \
        --random-input-len 2 \
        --random-output-len 1900 \
        --ignore-eos \
        >"${LOG_DIR}/saida_benchmark.txt" 2>&1

# 5. Cleanup: Kill the vLLM server once the benchmark is complete
echo "[INFO] Benchmark complete. Shutting down the vLLM server (PID: $SERVER_PID)..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "[INFO] Server shut down successfully."