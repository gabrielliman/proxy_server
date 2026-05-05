#!/bin/bash

set -e

ENV_NAME="vllm_env"
PYTHON_VERSION="3.12"

# Inicializa o Conda para este script (necessário para o comando 'conda activate' funcionar no bash)
eval "$(conda shell.bash hook)"

# Verifica se o ambiente Conda já existe
if ! conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Ambiente Conda '$ENV_NAME' não encontrado. Criando um novo..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
else
    echo "Ambiente Conda '$ENV_NAME' já existe. Ignorando a criação."
fi

echo "Ativando o ambiente Conda '$ENV_NAME'..."
conda activate "$ENV_NAME"

echo "Instalando vllm com uv..."
# O uv pip reconhece automaticamente o ambiente Conda ativo
uv pip install vllm --torch-backend=auto

echo "Configuração e instalação concluídas com sucesso!"


# export HF_HOME="/scratch/global/huggingface_cache/huggingface"

MODEL1_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL2_NAME="meta-llama/Llama-3.1-8B-Instruct"

MODEL1_PORT=8105
MODEL2_PORT=8106

MODEL1_MAX_NUM_SEQS=10
MODEL2_MAX_NUM_SEQS=10

LOG_DIR="./var/logs"

wait_for_ready() {
  local PORT=$1
  echo "[INFO] Waiting for model on port $PORT to become ready..."
  until curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; do
    sleep 5
  done
  echo "[INFO] Model on port $PORT is ready!"
}

export VLLM_SERVER_DEV_MODE=1


echo "[INFO] Starting vLLM inference servers"

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model "$MODEL1_NAME" --port $MODEL1_PORT --max-num-seqs $MODEL1_MAX_NUM_SEQS --dtype bfloat16 --max-model-len 40000 --gpu-memory-utilization 0.9  2>&1| tee "$LOG_DIR/saida_VLLM_$MODEL1_PORT.txt" &
PID1=$!
wait_for_ready $MODEL1_PORT

#Se quiser outra copia, so descomentar as linhas abaixo
# CUDA_VISIBLE_DEVICES=1 python -u -m vllm.entrypoints.openai.api_server --model "$MODEL2_NAME" --port $MODEL2_PORT --max-num-seqs $MODEL2_MAX_NUM_SEQS --dtype bfloat16 --max-model-len 40000 --gpu-memory-utilization 0.5 2>&1 | tee "$LOG_DIR/saida_VLLM_$MODEL2_PORT.txt" &
# PID2=$!
# wait_for_ready $MODEL2_PORT


wait $PID1 $PID2
