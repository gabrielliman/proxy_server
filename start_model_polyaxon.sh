#!/bin/bash


#Model Parameters
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH="/scratch/hpc4ai/models/llama/Llama-3.1-8B-Instruct"
MAX_NUM_SEQS=30
LOG_DIR="./var/logs"

# Number of instances parameters
NUM_INSTANCES=1       # Change this to the number of ports you want
START_PORT=8105        # The starting port number
GPU=0 #starting gpu
ENGINE_IDX=1 
PER_GPU=1
GPU_PERCENT=0.125

source /opt/conda/etc/profile.d/conda.sh && conda activate proxy_server
set -x

export MODEL="$MODEL_NAME"

# Initialize empty arrays
PORTS=()
ROUTES_LIST=()
PARALLELISM_LIST=()

# Loop to generate the ports and JSON elements
for (( i=0; i<NUM_INSTANCES; i++ )); do
  PORT=$((START_PORT + i))
  PORTS+=("$PORT")
  ROUTES_LIST+=("\"http://localhost:$PORT\"")
  PARALLELISM_LIST+=("\"http://localhost:$PORT\": $PARALLELISM_VALUE")
done


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


VLLM_PIDS=()
# cleanup_engines() {
#     echo "[CLEANUP] Automatically terminating vLLM engines..."
#     if [ ${#VLLM_PIDS[@]} -gt 0 ]; then
#         kill "${VLLM_PIDS[@]}" 2>/dev/null || true
#     fi
# }
# # Trap normal exits (0), Ctrl+C (2), and error exits (15)
# trap cleanup_engines EXIT INT TERM


# Loop through the PORTS array and start a server for each
for PORT in "${PORTS[@]}"; do
    echo "[INFO] Booting server on port $PORT..."
    
    # Your exact pipeline syntax
    CUDA_VISIBLE_DEVICES=$GPU python -u -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --dtype bfloat16 \
        --max-model-len 9000 \
        --gpu-memory-utilization $GPU_PERCENT \
        --served-model-name $MODEL_NAME \
        > >(tee "$LOG_DIR/saida_VLLM_${PORT}.txt") 2>&1 &
        
    # This executes exactly as: PID1=$!, PID2=$!, etc.
    VLLM_PIDS+=($!)
    
    # Print verification showing the variable name and value
    eval "echo '[INFO] Engine $PORT assigned to PID${ENGINE_IDX}=\$PID${ENGINE_IDX}'"
        
    wait_for_ready "$PORT"
    if (( ENGINE_IDX % $PER_GPU == 0 )); then
        GPU=$((GPU + 1))
    fi
    ENGINE_IDX=$((ENGINE_IDX + 1))
done


echo "======================================"
echo "Teste completo"
echo "======================================"
