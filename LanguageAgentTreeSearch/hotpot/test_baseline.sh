#!/bin/bash

# Criação dos diretórios de log e resultados (caso não existam)
export CUSTOM_API_BASE="http://localhost:8081"
export CUSTOM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
# Parâmetros base compartilhados
ALGORITHM="lats"
START_INDEX=900
END_INDEX=902
ITERATIONS=3
N_GENERATE=10
N_EVALUATE=1

# Nomenclatura compatível com a regex do get_baseline_metrics
# Formato esperado: baseline_output_{sched}_round-robin_rate{rate}_run{x}.json
BASELINE_JSON="/mnt/scratch/scheduler/proxy_server/LanguageAgentTreeSearch/hotpot/results/baseline_output_${ALGORITHM}_round-robin_rate10_run1.json"
NORMAL_JSON="/mnt/scratch/scheduler/proxy_server/LanguageAgentTreeSearch/hotpot/results/output_${ALGORITHM}_teste_rate10_run1.json"

echo "=========================================================="
echo "Iniciando Execução BASELINE (Calculando Thresholds)"
echo "=========================================================="

python run.py \
  --algorithm $ALGORITHM \
  --task_start_index $START_INDEX \
  --task_end_index $END_INDEX \
  --iterations $ITERATIONS \
  --log logs/lats_smoke_test_baseline.log \
  --n_generate_sample $N_GENERATE \
  --n_evaluate_sample $N_EVALUATE \
  --output-json $BASELINE_JSON \
  --is_baseline_run 1

echo -e "\n=========================================================="
echo "Iniciando Execução NORMAL (Usando Thresholds do Baseline)"
echo "=========================================================="

python run.py \
  --algorithm $ALGORITHM \
  --task_start_index $START_INDEX \
  --task_end_index $END_INDEX \
  --iterations $ITERATIONS \
  --log logs/lats_smoke_test_normal.log \
  --n_generate_sample $N_GENERATE \
  --n_evaluate_sample $N_EVALUATE \
  --output-json $NORMAL_JSON \
  --is_baseline_run 0

echo -e "\n=========================================================="
echo "Benchmarks Concluídos!"
echo "Baseline JSON: $BASELINE_JSON"
echo "Normal JSON:   $NORMAL_JSON"
echo "=========================================================="