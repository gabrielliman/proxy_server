#!/bin/bash

ulimit -n 524288

#Model Parameters
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
MAX_NUM_SEQS=192
LOG_DIR="./var/logs"

# Number of instances parameters
NUM_INSTANCES=1       # Change this to the number of ports you want
START_PORT=8106       # The starting port number
PARALLELISM_VALUE=10000   # The parallelism value for all ports # changed for waiting to have effect
GPU=1 #starting gpu
ENGINE_IDX=1  # Counter to create PID1, PID2, etc.
PER_GPU=1
GPU_PERCENT=0.97
MAX_MODEL_LEN=5000 #40000

# Lats parameters
export PROXY_PORT=8081
BASE_URL="http://localhost:${PROXY_PORT}" #proxy server url
ALGORITHM="lats"
END_INDEX=1000 #padrao: 100
ITERATIONS=10 #padrao: 50
N_GENERATE=5  #padrao: 5
N_EVALUATE=1  #padrao: 1
DEPTH=7       #padrao: 7
export ENABLE_ADMISSION_CONTROL="True"  # Altere para "False" para desligar
export ADMISSION_WINDOW_S="10.0"        # Janela da média móvel em segundos
export AIMD_LOG_CSV="./var/logs/aimd_decisions.csv"  # Caminho para o log de decisões do AIMD

# NOVO: Lista com o número de programas/tarefas para variar automaticamente
NUM_PROGRAMS_LIST=(304) 
# NUM_PROGRAMS_LIST=(19 38 76 114 152 190 228 266 304 342 380 418 456 494 532 570 608 646 684 722 760 798 836 874 912 950) 

# Benchmark parameters
OUTPUTS_DIR="outputs_lats/304prog_10it_new"
REPEATS=1
RATES=("inf")
BURSTINESS=1
BASELINE_SCHEDULER=autellix
EXPERIMENT_CONFIGS=(
    #ATLAS
    # "atlas:service_cumulative:autellix:N/A:2048"
    #"atlas:service_cumulative:least-total-load:N/A:100"
    #baseline autellix
    #"atlas:service_cumulative:autellix:N/A:0"
    # "atlas:service_cumulative:threshold-autellix:waiting:15"
    # "atlas:service_cumulative:ordered-dynamic-autellix:N/A:100"
    # "atlas:service_cumulative:least-load-dynamic-autellix:N/A:0"
    # "atlas:service_cumulative:probabilistic-cascade-autellix:N/A:100"
    #"atlas:service_cumulative:ordered-dynamic-autellix-reorder:N/A:100"
    #"atlas:service_cumulative:probabilistic-cascade-autellix-reorder:N/A:100"

    #FCFS
    # "fcfs:N/A:autellix:N/A:2048"
    # "fcfs:N/A:least-total-load:N/A:100"
    #"fcfs:N/A:autellix:N/A:0"
    # "fcfs:N/A:threshold-autellix:waiting:15"
    # "fcfs:N/A:ordered-dynamic-autellix:N/A:100"
    # "fcfs:N/A:least-load-dynamic-autellix:N/A:0"
    # "fcfs:N/A:probabilistic-cascade-autellix:N/A:100"
    # "fcfs:N/A:ordered-dynamic-autellix-reorder:N/A:100"
    # "fcfs:N/A:probabilistic-cascade-autellix-reorder:N/A:100"
)

SCHEDULER_CONFIGS=(
    "fcfs:N/A"
    # "plas:service_cumulative"
    # "plas:kv_token_time"
    # "atlas:service_cumulative"
    # "atlas:kv_token_time"
)

mkdir -p "$OUTPUTS_DIR"
export HF_HOME="/mnt/scratch/global/huggingface_cache/huggingface"
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /mnt/scratch/scheduler/envs/proxy_server
set -x

# ## arruma um dos warning, mas nao entendi direito
FLASHINFER_DIR="/opt/conda/envs/proxy_server/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/comm"

echo "[INFO] Verificando e corrigindo headers do FlashInfer..."
for file in "trtllm_allreduce_fusion.cuh" "trtllm_moe_allreduce_fusion.cuh"; do
    if [ -f "$FLASHINFER_DIR/$file" ]; then
        if ! grep -q "#include <optional>" "$FLASHINFER_DIR/$file"; then
            echo "[INFO] Aplicando hotfix em $file..."
            sed -i '1s/^/#include <optional>\n/' "$FLASHINFER_DIR/$file"
        else
            echo "[INFO] $file já está corrigido."
        fi
    fi
done

rm ./kv_cache_usage.csv 2>/dev/null || true

# Define the ports as an array. Add or remove ports here to automatically scale.
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

# Join the arrays with commas
ROUTES_JOINED=$(IFS=,; echo "${ROUTES_LIST[*]}")
PARALLELISM_JOINED=$(IFS=,; echo "${PARALLELISM_LIST[*]}")

# Construct and export the final JSON strings
export MODEL_ROUTES="{
  \"$MODEL_NAME\": [
    $ROUTES_JOINED
  ]
}"

export BACKEND_PARALLELISM="{
  $PARALLELISM_JOINED
}"

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
cleanup_engines() {
    echo "[CLEANUP] Automatically terminating vLLM engines..."
    if [ ${#VLLM_PIDS[@]} -gt 0 ]; then
        kill "${VLLM_PIDS[@]}" 2>/dev/null || true
    fi
}
# Trap normal exits (0), Ctrl+C (2), and error exits (15)
trap cleanup_engines EXIT INT TERM

# Loop through the PORTS array and start a server for each
for PORT in "${PORTS[@]}"; do
    echo "[INFO] Booting server on port $PORT..."
    
    CUDA_VISIBLE_DEVICES=$GPU python -u -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --dtype bfloat16 \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_PERCENT \
        --served-model-name $MODEL_NAME \
        > >(tee "$LOG_DIR/saida_VLLM_${PORT}.txt") 2>&1 &
        
    VLLM_PIDS+=($!)
    eval "echo '[INFO] Engine $PORT assigned to PID${ENGINE_IDX}=\$PID${ENGINE_IDX}'"
        
    wait_for_ready "$PORT"
    if (( ENGINE_IDX % $PER_GPU == 0 )); then
        GPU=$((GPU + 1))
    fi
    ENGINE_IDX=$((ENGINE_IDX + 1))
done

# ============================================================
# Benchmark Setup
# ============================================================
export CUSTOM_API_BASE="$BASE_URL"
export CUSTOM_MODEL="$MODEL_NAME"

# ============================================================
# Engines (metrics + reset) dynamically generated from PORTS
# ============================================================
RESET_URLS=()
PREFIX_URL=()

for PORT in "${PORTS[@]}"; do
    RESET_URLS+=("http://localhost:${PORT}/reset_prefix_cache")
    PREFIX_URL+=("http://localhost:${PORT}/metrics")
done

# ============================================================
# Helper: scrape per-engine metrics
# ============================================================
get_prefix_metrics_per_engine() {
    local url=$1
    local port=$2 

    curl -s "$url" | awk -v port="$port" '
    /^vllm:prefix_cache_queries_total\{/ {
        if (match($0, /engine="[^"]+"/)) {
            engine_str = substr($0, RSTART, RLENGTH)
            split(engine_str, parts, /"/)
            engine = parts[2]
            queries[engine] = $NF
        }
    }

    /^vllm:prefix_cache_hits_total\{/ {
        if (match($0, /engine="[^"]+"/)) {
            engine_str = substr($0, RSTART, RLENGTH)
            split(engine_str, parts, /"/)
            engine = parts[2]
            hits[engine] = $NF
        }
    }

    END {
        for (e in queries) {
            q = queries[e] + 0
            h = (e in hits ? hits[e] : 0) + 0
            printf "%s_%s %f %f\n", port, e, q, h
        }
    }'
}

# ============================================================
# baseline LOOP
# ============================================================
for NUM_PROGRAMS in "${NUM_PROGRAMS_LIST[@]}"; do
    START_INDEX=$((END_INDEX - NUM_PROGRAMS))
    
    echo "======================================"
    echo "[BASELINE] Testando com NUM_PROGRAMS = $NUM_PROGRAMS (START_INDEX=$START_INDEX)"
    echo "======================================"

    for sched_conf in "${SCHEDULER_CONFIGS[@]}"; do
        IFS=':' read -r scheduler plas_metric <<< "$sched_conf"

        sched_suffix=$scheduler
        if [ "$scheduler" = "plas" ] || [ "$scheduler" = "atlas" ]; then
            sched_suffix="${scheduler}_${plas_metric}"
        fi

        for rate in "${RATES[@]}"; do
            for RUN in $(seq 1 $REPEATS); do
                echo "Run $RUN / $REPEATS [Baseline] - Progs: $NUM_PROGRAMS"
                PROXY_ERROR="${OUTPUTS_DIR}/ERROR_proxy_output_${sched_suffix}_autellix_rate${rate}_${NUM_PROGRAMS}progs_run${RUN}.txt"

                SCHEDULER="$scheduler" \
                PLAS_METRIC_TYPE="$plas_metric" \
                LOAD_BALANCER_STRATEGY="autellix" \
                python main.py 2> "$PROXY_ERROR" &
                PROXY_PID=$!
                sleep 10 

                for url in "${RESET_URLS[@]}"; do
                    curl -s -X POST "$url" > /dev/null
                done
                sleep 2

                unset before_q before_h prefix_json_map
                declare -A before_q
                declare -A before_h

                for PORT in "${PORTS[@]}"; do
                    url="http://localhost:${PORT}/metrics"
                    while read -r engine q h; do
                        before_q[$engine]=$q
                        before_h[$engine]=$h
                    done < <(get_prefix_metrics_per_engine "$url" "$PORT")
                done
                
                OUTPUT_JSON="${OUTPUTS_DIR}/baseline_output_${sched_suffix}_autellix_rate${rate}_${NUM_PROGRAMS}progs_run${RUN}.json"
                OUTPUT_ERROR="${OUTPUTS_DIR}/ERROR_baseline_output_${sched_suffix}_autellix_rate${rate}_${NUM_PROGRAMS}progs_run${RUN}.txt"
                OUTPUT_KV_CACHE="${OUTPUTS_DIR}/baseline_output_${sched_suffix}_autellix_rate${rate}_${NUM_PROGRAMS}progs_run${RUN}_kv_cache.csv"
                OUTPUT_ADMISSION_LOG="${OUTPUTS_DIR}/baseline_output_${sched_suffix}_autellix_rate${rate}_${NUM_PROGRAMS}progs_run${RUN}_admissionlog.csv"

                python LanguageAgentTreeSearch/hotpot/run.py \
                    --algorithm $ALGORITHM \
                    --task_start_index $START_INDEX \
                    --task_end_index $END_INDEX \
                    --iterations $ITERATIONS \
                    --n_generate_sample $N_GENERATE \
                    --n_evaluate_sample $N_EVALUATE \
                    --output-json $OUTPUT_JSON \
                    --burstiness $BURSTINESS \
                    --program-rate "$rate" \
                    --depth_limit "$DEPTH" \
                    --is_baseline_run 1 2> "$OUTPUT_ERROR"
                    
                PROG_FILE="${OUTPUTS_DIR}/baseline_processes_summary_${sched_suffix}_autellix_rate${rate}_${NUM_PROGRAMS}progs_run${RUN}.json"
                curl -s "${BASE_URL}/processes_summary" | python3 -m json.tool > "$PROG_FILE"
                
                echo "Stopping proxy (PID=$PROXY_PID)"
                kill "$PROXY_PID"
                wait "$PROXY_PID" 2>/dev/null || true
                mv ./kv_cache_usage.csv "$OUTPUT_KV_CACHE"
                mv AIMD_LOG_CSV "$OUTPUT_ADMISSION_LOG"
                sleep 5

                declare -A prefix_json_map

                for PORT in "${PORTS[@]}"; do
                    url="http://localhost:${PORT}/metrics"
                    while read -r engine q h; do
                        before_queries=${before_q[$engine]:-0}
                        before_hits=${before_h[$engine]:-0}

                        dq=$(awk -v q="${q:-0}" -v bq="${before_queries:-0}" 'BEGIN { print q - bq }')
                        dh=$(awk -v h="${h:-0}" -v bh="${before_hits:-0}" 'BEGIN { print h - bh }')
                        hr=$(awk -v dh="${dh:-0}" -v dq="${dq:-0}" 'BEGIN { if (dq>0) printf "%.6f", dh/dq; else print 0 }')

                        prefix_json_map[$engine]="{\"queries\":${dq:-0},\"hits\":${dh:-0},\"hit_rate\":${hr:-0}}"
                    done < <(get_prefix_metrics_per_engine "$url" "$PORT")
                done

                prefix_json="{"
                first=1
                for engine in "${!prefix_json_map[@]}"; do
                    if [ $first -eq 0 ]; then
                        prefix_json+=","
                    fi
                    prefix_json+="\"engine_${engine}\":${prefix_json_map[$engine]}"
                    first=0
                done
                prefix_json+="}"

                python3 -c '
import sys, json

file_path = sys.argv[1]
prefix_data = json.loads(sys.argv[2])
scheduler = sys.argv[3]
strategy = sys.argv[4]
rate = float(sys.argv[5])
start = int(sys.argv[6])
end = int(sys.argv[7])

with open(file_path, "r") as f:
    data = json.load(f)

if "full_metrics" not in data:
    data["full_metrics"] = {}

data["full_metrics"]["prefix_cache"] = prefix_data
data["full_metrics"]["scheduler"] = scheduler
data["full_metrics"]["load_balancer"] = strategy
data["full_metrics"]["request_rate"] = rate
data["full_metrics"]["num_conversations"] = end-start

with open(file_path, "w") as f:
    json.dump(data, f, indent=2)
' "$OUTPUT_JSON" "$prefix_json" "$sched_suffix" "autellix" "$rate" "$START_INDEX" "$END_INDEX"

                echo "Metrics injected into JSON"
                echo "--------------------------------------"
            done
        done
    done
done
