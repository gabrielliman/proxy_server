#!/bin/bash
set -a
source "/mnt/scratch/scheduler/proxy_server/.env"
export HF_HOME="/mnt/scratch/global/huggingface_cache/huggingface"

# Paths and ports
MODEL1_NAME="/mnt/scratch/global/huggingface_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
MODEL2_NAME="meta-llama/Llama-3.1-8B-Instruct"
# MODEL3_NAME="meta-llama/Llama-3.1-8B-Instruct"
# "meta-llama/Llama-3.1-8B-Instruct"
# "Qwen/Qwen2.5-1.5B-Instruct"
# "Qwen/Qwen2.5-0.5B-Instruct"
# "Qwen/Qwen3-0.6B"
# "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

EMBEDDING_PORT=8001
MODEL1_PORT=8105
MODEL2_PORT=8106
#MODEL3_PORT=8007


MODEL1_MAX_NUM_SEQS=10
MODEL2_MAX_NUM_SEQS=10

LOG_DIR="/mnt/scratch/scheduler/proxy_server/var/logs"

source /opt/miniconda/etc/profile.d/conda.sh
conda activate /mnt/scratch/scheduler/envs/proxy_server


wait_for_ready() {
  local PORT=$1
  echo "[INFO] Waiting for model on port $PORT to become ready..."
  until curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; do
    sleep 5
  done
  echo "[INFO] Model on port $PORT is ready!"
}


# echo "[INFO] Starting vLLM embedding server"
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model nomic-ai/nomic-embed-text-v1 --task embed --port $EMBEDDING_PORT --trust-remote-code --max-model-len 8K &
# PIDEmbedding=$!
# wait_for_ready $EMBEDDING_PORT
export VLLM_SERVER_DEV_MODE=1


echo "[INFO] Starting vLLM inference servers"

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model "$MODEL1_NAME" --port $MODEL1_PORT --max-num-seqs $MODEL1_MAX_NUM_SEQS --dtype bfloat16 --max-model-len 40000 --gpu-memory-utilization 0.9  2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL1_PORT.txt" &
PID1=$!
wait_for_ready $MODEL1_PORT

# CUDA_VISIBLE_DEVICES=0 python -u -m vllm.entrypoints.openai.api_server --model "$MODEL2_NAME" --port $MODEL2_PORT --max-num-seqs $MODEL2_MAX_NUM_SEQS --dtype bfloat16 --max-model-len 40000 --gpu-memory-utilization 0.9 2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL2_PORT.txt" &
# PID2=$!
# wait_for_ready $MODEL2_PORT




# CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --tensor-parallel-size 2 --model "$MODEL3_NAME" --port $MODEL3_PORT --dtype bfloat16 --max-model-len 20000 --gpu-memory-utilization 0.3  2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL3_PORT.txt" &
# PID3=$!
# wait_for_ready $MODEL3_PORT




wait $PID1 $PID2 # $PIDEmbedding