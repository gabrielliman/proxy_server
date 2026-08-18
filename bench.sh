#!/bin/bash

# Define environment variables
export HF_HOME="/mnt/scratch/global/huggingface_cache/huggingface"
export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export NUM_PROMPTS=1000 
export REQUEST_RATE="inf"

# Define the health-check function
wait_for_ready() {
  local PORT=$1
  echo "[INFO] Waiting for model on port $PORT to become ready..."
  until curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; do
    sleep 5
  done
  echo "[INFO] Model on port $PORT is ready!"
}

# 1. Start the server in the background (Notice the trailing '&' and removed '\')
CUDA_VISIBLE_DEVICES=0 python -u -m vllm.entrypoints.openai.api_server \
        --model "meta-llama/Llama-3.1-8B-Instruct" \
        --port "8000" \
        --max-num-seqs "1024" \
        --dtype bfloat16 \
        --max-model-len 10000 \
        --gpu-memory-utilization 0.9 \
        --served-model-name "meta-llama/Llama-3.1-8B-Instruct" &

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
        --ignore-eos

# 5. Cleanup: Kill the vLLM server once the benchmark is complete
echo "[INFO] Benchmark complete. Shutting down the vLLM server (PID: $SERVER_PID)..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "[INFO] Server shut down successfully."



# ============ Serving Benchmark Result ============
# Successful requests:                     1000      
# Failed requests:                         0         
# Benchmark duration (s):                  1256.13   
# Total input tokens:                      2000      
# Total generated tokens:                  1900000   
# Request throughput (req/s):              0.80      
# Output token throughput (tok/s):         1512.59   
# Peak output token throughput (tok/s):    8448.00   
# Peak concurrent requests:                1000.00   
# Total token throughput (tok/s):          1514.18   
# ---------------Time to First Token----------------
# Mean TTFT (ms):                          422680.66 
# Median TTFT (ms):                        327314.25 
# P50 TTFT (ms):                           327314.25 
# P75 TTFT (ms):                           701534.18 
# P90 TTFT (ms):                           909597.73 
# P99 TTFT (ms):                           1036863.78
# -----Time per Output Token (excl. 1st token)------
# Mean TPOT (ms):                          116.63    
# Median TPOT (ms):                        122.63    
# P50 TPOT (ms):                           122.63    
# P75 TPOT (ms):                           142.90    
# P90 TPOT (ms):                           159.65    
# P99 TPOT (ms):                           172.02    
# ---------------Inter-token Latency----------------
# Mean ITL (ms):                           116.64    
# Median ITL (ms):                         24.74     
# P50 ITL (ms):                            24.74     
# P75 ITL (ms):                            25.29     
# P90 ITL (ms):                            27.18     
# P99 ITL (ms):                            183.62    
# ----------------End-to-end Latency----------------
# Mean E2EL (ms):                          644165.28 
# Median E2EL (ms):                        646867.79 
# P50 E2EL (ms):                           646867.79 
# P75 E2EL (ms):                           951688.06 
# P90 E2EL (ms):                           1146599.89
# P99 E2EL (ms):                           1240326.86
# ==================================================