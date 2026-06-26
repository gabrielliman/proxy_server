#!/bin/bash
# START VLLM SERVERS

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

# ============================================================
# Global Configuration
# ============================================================

# Set the model name once
rm ./kv_cache_usage.csv 2>/dev/null || true

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH="/scratch/hpc4ai/models/llama/Llama-3.1-8B-Instruct"


# Define the ports as an array. Add or remove ports here to automatically scale.
PORTS=(8105 8106 8107 8108)

# Shared parameters
MAX_NUM_SEQS=256
LOG_DIR="./var/logs"

export MODEL="$MODEL_NAME"
export MODEL_ROUTES="{
  \"$MODEL_NAME\": [
    \"http://localhost:8105\", 
    \"http://localhost:8106\",
    \"http://localhost:8107\",
    \"http://localhost:8108\"
  ]
}"
export BACKEND_PARALLELISM='{
  "http://localhost:8105": 256, 
  "http://localhost:8106": 256,
  "http://localhost:8107": 256,
  "http://localhost:8108": 256
}'

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
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --dtype bfloat16 \
        --max-model-len 40000 \
        --gpu-memory-utilization 0.45 \
        --served-model-name $MODEL_NAME \
        2>&1 | tee "$LOG_DIR/saida_VLLM_${PORT}.txt" &
        
    # This executes exactly as: PID1=$!, PID2=$!, etc.
    eval "PID${ENGINE_IDX}=\$!"
    
    # Print verification showing the variable name and value
    eval "echo '[INFO] Engine $PORT assigned to PID${ENGINE_IDX}=\$PID${ENGINE_IDX}'"
        
    wait_for_ready "$PORT"
    if (( ENGINE_IDX % 2 == 0 )); then
        GPU=$((GPU + 1))
    fi
    ENGINE_IDX=$((ENGINE_IDX + 1))
done
# ============================================================
# Benchmark Setup
# ============================================================
BASE_URL="http://localhost:8081"
#trocar para local na scratch
DATASET="./ShareGPT_V3_unfiltered_cleaned_split.json"

if [ ! -f "$DATASET" ]; then
    echo "[INFO] Downloading dataset..."
    curl -L https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -o "$DATASET"
fi

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
MODEL="$MODEL_NAME"
LIMIT=66400 #numero de conversas do share gpt (num programas)
OUTPUTS_DIR="outputs/teste"
mkdir -p "$OUTPUTS_DIR"
CHAT_LEN=-1 #tamanho fixo das conversas, se -1 pega de qualquer
REPEATS=1
RATES=("1000")
# RATES=("0.5" "1" "2" "4" "8")
BURSTINESS=1

# Format: "scheduler_name:plas_metric_type"
SCHEDULER_CONFIGS=(
  "fcfs:N/A"
#   "plas:service_cumulative"
#   "plas:kv_token_time"
) 

# Format: "strategy_name:threshold_metric:threshold_value"
LB_CONFIGS=(
  "least-total-load:N/A:0"
#   "least-waiting:N/A:0"
#   "least-running:N/A:0"
#   "least-kv-cache:N/A:0"
#   "autellix:N/A:0"
#   "threshold-autellix:total:15"
#   "threshold-autellix:running:10"
#   "threshold-autellix:kv_cache_percent:90"
)

# ============================================================
# Helper: scrape per-engine metrics
# ============================================================

get_prefix_metrics_per_engine() {
    local url=$1

    curl -s "$url" | awk '

    /^vllm:prefix_cache_queries_total\{/ {
        # Standard 2-arg match finds the starting position (RSTART) and length (RLENGTH)
        if (match($0, /engine="[^"]+"/)) {
            # Extract exactly engine="<name>"
            engine_str = substr($0, RSTART, RLENGTH)
            # Split by double quotes to get the actual name
            split(engine_str, parts, /"/)
            engine = parts[2]
            
            # In Prometheus metrics, the value is always the last field ($NF)
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

            # MACHINE READABLE (important!)
            printf "%s %f %f\n", e, q, h
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
        SCHEDULER="$scheduler" \
        PLAS_METRIC_TYPE="$plas_metric" \
        LOAD_BALANCER_STRATEGY="round-robin" \
        python main.py &
        PROXY_PID=$!
        sleep 10  # tempo para proxy + engines estabilizarem
        for RUN in $(seq 1 $REPEATS); do

            echo "Run $RUN / $REPEATS [Baseline]"

            for url in "${RESET_URLS[@]}"; do
                curl -s -X POST "$url" > /dev/null
            done
            sleep 2

            unset before_q before_h prefix_json_map
            declare -A before_q
            declare -A before_h

            for url in "${PREFIX_URL[@]}"; do
                while read -r engine q h; do
                    before_q[$engine]=$q
                    before_h[$engine]=$h
                done < <(get_prefix_metrics_per_engine "$url")
            done
            OUTPUT_JSON="${OUTPUTS_DIR}/baseline_output_${sched_suffix}_round-robin_rate${rate}_run${RUN}.json"
            OUTPUT_KV_CACHE="${OUTPUTS_DIR}/baseline_output_${sched_suffix}_round-robin_rate${rate}_run${RUN}_kv_cache.csv"
            python benchmark_stateful.py \
                --base-url "$BASE_URL" \
                --dataset "$DATASET" \
                --model "$MODEL" \
                --limit "$LIMIT" \
                --request-rate "$rate" \
                --mode "chat" \
                --output-json "$OUTPUT_JSON" \
                --chat_len "$CHAT_LEN" \
                --is_baseline_run 1 \
                --burstiness $BURSTINESS
        done
        #Matando o proxy
        echo "Stopping proxy (PID=$PROXY_PID)"
        kill "$PROXY_PID"
        wait "$PROXY_PID" 2>/dev/null || true
        mv ./kv_cache_usage.csv "$OUTPUT_KV_CACHE"
        sleep 5

        declare -A prefix_json_map

        for url in "${PREFIX_URL[@]}"; do
            while read engine q h; do
                before_queries=${before_q[$engine]:-0}
                before_hits=${before_h[$engine]:-0}
                dq=$(echo "$q - $before_queries" | bc)
                dh=$(echo "$h - $before_hits" | bc)
                hr=$(awk -v dh="$dh" -v dq="$dq" 'BEGIN { if (dq>0) printf "%.6f", dh/dq; else print 0 }')
                prefix_json_map[$engine]="{\"queries\":$dq,\"hits\":$dh,\"hit_rate\":$hr}"
            done < <(get_prefix_metrics_per_engine "$url")
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
        tmp=$(mktemp)

        jq \
        --argjson prefix "$prefix_json" \
        --arg scheduler "$sched_suffix" \
        --arg strategy "round-robin" \
        --arg rate "$rate" \
        --arg num_conversations "$LIMIT" \
        '
        .full_metrics.prefix_cache = $prefix
        | .full_metrics.scheduler = $scheduler
        | .full_metrics.load_balancer = $strategy
        | .full_metrics.request_rate = ($rate | tonumber)
        | .full_metrics.num_conversations = ($num_conversations | tonumber)
        ' "$OUTPUT_JSON" > "$tmp" && mv "$tmp" "$OUTPUT_JSON"

        echo "Metrics injected into JSON"
        echo "--------------------------------------"

    done
done

# ============================================================
# MAIN LOOP
# ============================================================

for sched_conf in "${SCHEDULER_CONFIGS[@]}"; do
    IFS=':' read -r scheduler plas_metric <<< "$sched_conf"

    sched_suffix=$scheduler
    if [ "$scheduler" = "plas" ] || [ "$scheduler" = "atlas" ]; then
        sched_suffix="${scheduler}_${plas_metric}"
    fi

    for rate in "${RATES[@]}"; do

        for lb_conf in "${LB_CONFIGS[@]}"; do
            IFS=':' read -r strategy thresh_metric thresh_val <<< "$lb_conf"

            # Create a string for JSON filename
            strat_suffix=$strategy
            if [ "$strategy" == "threshold-autellix" ]; then
                strat_suffix="${strategy}_${thresh_metric}_${thresh_val}"
            fi

            echo "======================================"
            echo "Starting proxy:"
            echo "  Scheduler = $scheduler (PLAS: $plas_metric)"
            echo "  LB        = $strategy (Thresh: $thresh_metric=$thresh_val)"
            echo "  Rate      = $rate"
            echo "======================================"

            # ------------------------------------
            # Start proxy server
            # ------------------------------------
            SCHEDULER="$scheduler" \
            PLAS_METRIC_TYPE="$plas_metric" \
            LOAD_BALANCER_STRATEGY="$strategy" \
            LOAD_BALANCER_THRESHOLD_METRIC="$thresh_metric" \
            LOAD_BALANCER_THRESHOLD_VALUE="$thresh_val" \
            LOAD_BALANCER_FALLBACK_STRATEGY="least-total-load" \
            python main.py &

            PROXY_PID=$!
            sleep 10  # tempo para proxy + engines estabilizarem

            # ------------------------------------
            # Benchmark runs
            # ------------------------------------
            for RUN in $(seq 1 $REPEATS); do

                echo "Run $RUN / $REPEATS"

                for url in "${RESET_URLS[@]}"; do
                  curl -s -X POST "$url" > /dev/null
                done
                sleep 2

                unset before_q before_h prefix_json_map
                declare -A before_q
                declare -A before_h

                for url in "${PREFIX_URL[@]}"; do
                    while read -r engine q h; do
                        before_q[$engine]=$q
                        before_h[$engine]=$h
                    done < <(get_prefix_metrics_per_engine "$url")
                done
                OUTPUT_JSON="${OUTPUTS_DIR}/output_${sched_suffix}_${strat_suffix}_rate${rate}_run${RUN}.json"
                OUTPUT_KV_CACHE="${OUTPUTS_DIR}/output_${sched_suffix}_${strat_suffix}_rate${rate}_run${RUN}_kv_cache.csv"
                python benchmark_stateful.py \
                    --base-url "$BASE_URL" \
                    --dataset "$DATASET" \
                    --model "$MODEL" \
                    --limit "$LIMIT" \
                    --request-rate "$rate" \
                    --mode "chat" \
                    --output-json "$OUTPUT_JSON" \
                    --chat_len "$CHAT_LEN" \
                    --is_baseline_run 0 \
                    --burstiness $BURSTINESS

                # ------------------------------------
                # AFTER metrics + compute delta
                # ------------------------------------
                declare -A prefix_json_map

                for url in "${PREFIX_URL[@]}"; do
                    while read engine q h; do

                        before_queries=${before_q[$engine]:-0}
                        before_hits=${before_h[$engine]:-0}

                        dq=$(echo "$q - $before_queries" | bc)
                        dh=$(echo "$h - $before_hits" | bc)

                        hr=$(awk -v dh="$dh" -v dq="$dq" 'BEGIN { if (dq>0) printf "%.6f", dh/dq; else print 0 }')

                        prefix_json_map[$engine]="{\"queries\":$dq,\"hits\":$dh,\"hit_rate\":$hr}"
                    done < <(get_prefix_metrics_per_engine "$url")
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
                tmp=$(mktemp)

                jq \
                --argjson prefix "$prefix_json" \
                --arg scheduler "$sched_suffix" \
                --arg strategy "$strat_suffix" \
                --arg rate "$rate" \
                --arg num_conversations "$LIMIT" \
                '
                .full_metrics.prefix_cache = $prefix
                | .full_metrics.scheduler = $scheduler
                | .full_metrics.load_balancer = $strategy
                | .full_metrics.request_rate = ($rate | tonumber)
                | .full_metrics.num_conversations = ($num_conversations | tonumber)
                ' "$OUTPUT_JSON" > "$tmp" && mv "$tmp" "$OUTPUT_JSON"

                echo "Metrics injected into JSON"
                echo "--------------------------------------"
                

            done

            # ------------------------------------
            # Stop proxy server
            # ------------------------------------
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

echo "[INFO] Stopping vLLM engines..."
kill "$PID1" "$PID2" "$PID3" "$PID4" 2>/dev/null || true