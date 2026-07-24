#!/bin/bash

ulimit -n 524288

#Model Parameters
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH="/scratch/hpc4ai/models/llama/Llama-3.1-8B-Instruct"
MAX_NUM_SEQS=10
LOG_DIR="./var/logs"

# Number of instances parameters
NUM_INSTANCES=16       # Change this to the number of ports you want
START_PORT=8105        # The starting port number
PARALLELISM_VALUE=15   # The parallelism value for all ports
GPU=0 #starting gpu
ENGINE_IDX=1  # Counter to create PID1, PID2, etc.
PER_GPU=4
GPU_PERCENT=0.15
# Lats parameters

BASE_URL="http://localhost:8081" #proxy server url
ALGORITHM="lats"
START_INDEX=900
END_INDEX=910
ITERATIONS=10
N_GENERATE=10 #5
N_EVALUATE=1
DEPTH=14

# Benchmark parameters
OUTPUTS_DIR="outputs_lats/atlas_service_500prog_10it_100gen_rate8_bur01_100thres_2048out_100parall_depth7"
REPEATS=1
RATES=("8")
# RATES=("0.5" "1" "2" "4" "8")
BURSTINESS=0.1
BASELINE_SCHEDULER=autellix
EXPERIMENT_CONFIGS=(
    #baseline fcfs
    # "atlas:service_cumulative:least-total-load:N/A:0"
    #baseline autellix
    # "atlas:service_cumulative:autellix:N/A:0"
    # "atlas:service_cumulative:threshold-autellix:total:15"
    # "atlas:service_cumulative:ordered-dynamic-autellix:N/A:0"
    # "atlas:service_cumulative:least-load-dynamic-autellix:N/A:0"
    # "atlas:service_cumulative:probabilistic-cascade-autellix:N/A:0"
    # "atlas:service_cumulative:ordered-dynamic-autellix-reorder:N/A:0"
    # "atlas:service_cumulative:probabilistic-cascade-autellix-reorder:N/A:0"

    #nossa proposta escalonador
    # "atlas:kv_token_time:autellix:N/A:0"
    #nossa proposta lb
    # "atlas:service_cumulative:threshold-autellix:running:10"
    # "atlas:service_cumulative:ordered-dynamic-autellix:N/A:0"
    # "atlas:service_cumulative:least-load-dynamic-autellix:N/A:0"
    #nossa proposta combinada
    # "atlas:kv_token_time:ordered-dynamic-autellix:N/A:0"
)

SCHEDULER_CONFIGS=(
    # "fcfs:N/A"
    # "plas:service_cumulative"
    # "plas:kv_token_time"
    "atlas:service_cumulative"
    # "atlas:kv_token_time"

)

source /opt/conda/etc/profile.d/conda.sh && conda activate proxy_server
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
    
    # Your exact pipeline syntax
    CUDA_VISIBLE_DEVICES=$GPU python -u -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --dtype bfloat16 \
        --max-model-len 40000 \
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
# Experiment grid
# ============================================================
mkdir -p "$OUTPUTS_DIR"

# ============================================================
# Helper: scrape per-engine metrics
# ============================================================

get_prefix_metrics_per_engine() {
    local url=$1
    local port=$2 # Accept the port number

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

            # Prepend the port to the engine to guarantee uniqueness
            printf "%s_%s %f %f\n", port, e, q, h
        }
    }'
}
# ============================================================
# baseline LOOP
# ============================================================

for sched_conf in "${SCHEDULER_CONFIGS[@]}"; do
    IFS=':' read -r scheduler plas_metric <<< "$sched_conf"

    # Create a clear string for the JSON filename
    sched_suffix=$scheduler
    if [ "$scheduler" = "plas" ] || [ "$scheduler" = "atlas" ]; then
        sched_suffix="${scheduler}_${plas_metric}"
    fi

    for rate in "${RATES[@]}"; do

        # ------------------------------------
        # Start proxy server
        # ------------------------------------
        for RUN in $(seq 1 $REPEATS); do
            echo "Run $RUN / $REPEATS [Baseline]"
            PROXY_ERROR="${OUTPUTS_DIR}/ERROR_proxy_output_${sched_suffix}_autellix_rate${rate}_run${RUN}.txt"

            SCHEDULER="$scheduler" \
            PLAS_METRIC_TYPE="$plas_metric" \
            LOAD_BALANCER_STRATEGY="autellix" \
            python main.py 2> "$PROXY_ERROR" &
            PROXY_PID=$!
            sleep 10  # tempo para proxy + engines estabilizarem

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
            OUTPUT_JSON="${OUTPUTS_DIR}/baseline_output_${sched_suffix}_autellix_rate${rate}_run${RUN}.json"
            OUTPUT_ERROR="${OUTPUTS_DIR}/ERROR_baseline_output_${sched_suffix}_autellix_rate${rate}_run${RUN}.txt"
            OUTPUT_KV_CACHE="${OUTPUTS_DIR}/baseline_output_${sched_suffix}_autellix_rate${rate}_run${RUN}_kv_cache.csv"

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
            #Matando o proxy
            PROG_FILE="${OUTPUTS_DIR}/baseline_processes_summary_${sched_suffix}_autellix_rate${rate}_run${RUN}.json"
            curl -s "${BASE_URL}/processes_summary" | python3 -m json.tool > "$PROG_FILE"
            echo "Stopping proxy (PID=$PROXY_PID)"
            kill "$PROXY_PID"
            wait "$PROXY_PID" 2>/dev/null || true
            mv ./kv_cache_usage.csv "$OUTPUT_KV_CACHE"
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

            # ------------------------------------
            # Build JSON object
            # ------------------------------------
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

            # ------------------------------------
            # Inject into benchmark JSON
            # ------------------------------------
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

# ============================================================
# MAIN LOOP
# ============================================================

for config in "${EXPERIMENT_CONFIGS[@]}"; do
    # Lê as 5 variáveis de uma vez
    IFS=':' read -r scheduler plas_metric strategy thresh_metric thresh_val <<< "$config"

    # Define o sufixo do scheduler
    sched_suffix=$scheduler
    if [ "$scheduler" = "plas" ] || [ "$scheduler" = "atlas" ]; then
        sched_suffix="${scheduler}_${plas_metric}"
    fi

    # Define o sufixo do load balancer
    strat_suffix=$strategy
    if [ "$strategy" == "threshold-autellix" ]; then
        strat_suffix="${strategy}_${thresh_metric}_${thresh_val}"
    fi

    for rate in "${RATES[@]}"; do
        
        for RUN in $(seq 1 $REPEATS); do
            echo "======================================"
            echo "Starting proxy:"
            echo "  Scheduler = $scheduler (PLAS: $plas_metric)"
            echo "  LB        = $strategy (Thresh: $thresh_metric=$thresh_val)"
            echo "  Rate      = $rate"
            echo "======================================"

            # ------------------------------------
            # Start proxy server
            # ------------------------------------
            PROXY_ERROR="${OUTPUTS_DIR}/ERROR_proxy_output_${sched_suffix}_${strat_suffix}_rate${rate}_run${RUN}.txt"

            SCHEDULER="$scheduler" \
            PLAS_METRIC_TYPE="$plas_metric" \
            LOAD_BALANCER_STRATEGY="$strategy" \
            LOAD_BALANCER_THRESHOLD_METRIC="$thresh_metric" \
            LOAD_BALANCER_THRESHOLD_VALUE="$thresh_val" \
            LOAD_BALANCER_FALLBACK_STRATEGY="least-total-load" \
            python main.py 2> "$PROXY_ERROR" &

            PROXY_PID=$!
            sleep 10  # tempo para proxy + engines estabilizarem

            # ------------------------------------
            # Benchmark runs
            # ------------------------------------
        
            echo "Run $RUN / $REPEATS"

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
                OUTPUT_ERROR="${OUTPUTS_DIR}/ERROR_output_${sched_suffix}_${strat_suffix}_rate${rate}_run${RUN}.txt"
                OUTPUT_JSON="${OUTPUTS_DIR}/output_${sched_suffix}_${strat_suffix}_rate${rate}_run${RUN}.json"
                OUTPUT_KV_CACHE="${OUTPUTS_DIR}/output_${sched_suffix}_${strat_suffix}_rate${rate}_run${RUN}_kv_cache.csv"
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
                    --is_baseline_run 0 2> "$OUTPUT_ERROR"
                # ------------------------------------
                # AFTER metrics + compute delta
                # ------------------------------------
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

                # ------------------------------------
                # Build JSON object
                # ------------------------------------
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

                # ------------------------------------
                # Inject into benchmark JSON
                # ------------------------------------
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
' "$OUTPUT_JSON" "$prefix_json" "$sched_suffix" "$strat_suffix" "$rate" "$START_INDEX" "$END_INDEX"
                # Stop proxy server
                # ------------------------------------
                PROG_FILE="${OUTPUTS_DIR}/processes_summary_${sched_suffix}_${strat_suffix}_rate${rate}_run${RUN}.json"
                curl -s "${BASE_URL}/processes_summary" | python3 -m json.tool > "$PROG_FILE"
                echo "Stopping proxy (PID=$PROXY_PID)"
                kill "$PROXY_PID"
                wait "$PROXY_PID" 2>/dev/null || true
                mv ./kv_cache_usage.csv "$OUTPUT_KV_CACHE"
                sleep 5
        done
    done
done
# mv ./kv_cache_usage.csv "$OUTPUTS_DIR/kv_cache_usage.csv"

echo "======================================"
echo "All benchmarks completed successfully."
echo "======================================"
