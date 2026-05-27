#!/bin/bash
# START VLLM SERVERS
source /opt/conda/etc/profile.d/conda.sh && conda activate proxy_server
set -x

# ============================================================
# Global Configuration
# ============================================================
export HF_TOKEN=""

# Set the model name once
MODEL_NAME="meta-llama/Llama-3.1-70B"

# Define the ports as an array. Add or remove ports here to automatically scale.
PORTS=(8110)

# Shared parameters
MAX_NUM_SEQS=10
LOG_DIR="./var/logs"

# ============================================================
# Setup & Helper Functions
# ============================================================
mkdir -p "$LOG_DIR"

wait_for_ready() {
  local PORT=$1
  echo "[INFO] Waiting for model on port $PORT to become ready..."
  until curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; do
    sleep 5
  done
  echo "[INFO] Model on port $PORT is ready!"
}

# ============================================================
# Start vLLM Servers
# ============================================================
export VLLM_SERVER_DEV_MODE=1
echo "[INFO] Starting vLLM inference servers"

# Loop through the PORTS array and start a server for each
GPU=0
ENGINE_IDX=1  # Counter to create PID1, PID2, etc.

for PORT in "${PORTS[@]}"; do
    echo "[INFO] Booting server on port $PORT..."
    
    # Your exact pipeline syntax
    CUDA_VISIBLE_DEVICES=$GPU python -u -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --port "$PORT" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --dtype bfloat16 \
        --max-model-len 40000 \
        --gpu-memory-utilization 0.95 \
        2>&1 | tee "$LOG_DIR/saida_VLLM_${PORT}.txt" &
        
    # This executes exactly as: PID1=$!, PID2=$!, etc.
    eval "PID${ENGINE_IDX}=\$!"
    
    # Print verification showing the variable name and value
    eval "echo '[INFO] Engine $PORT assigned to PID${ENGINE_IDX}=\$PID${ENGINE_IDX}'"
        
    wait_for_ready "$PORT"
    # GPU=$((GPU + 1))
    ENGINE_IDX=$((ENGINE_IDX + 1))
done

sleep 100
kill $PID1