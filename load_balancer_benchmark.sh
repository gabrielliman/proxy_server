#!/bin/bash
set -e

# ============================================================
# Environment
# ============================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/global/abacus

BASE_URL="http://localhost:8081"
DATASET="/scratch/global/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"



MODEL="meta-llama/Llama-3.1-8B-Instruct"
LIMIT=50
OUTPUTS_DIR="outputs_llama_50_2gpu_burstiness_015"
mkdir -p "$OUTPUTS_DIR"

# ============================================================
# Engines (metrics + reset)
# OBS: benchmark coleta métricas apenas de UM endpoint
# ============================================================
RESET_URLS=(
  "http://localhost:8105/reset_prefix_cache"
  "http://localhost:8106/reset_prefix_cache"
)

PREFIX_URL=(
  "http://localhost:8105/metrics"
  "http://localhost:8106/metrics"
)


# ============================================================
# Experiment grid
# ============================================================
REPEATS=3

SCHEDULERS=("fcfs" "plas") 
LOAD_BALANCER_STRATEGIES=(
  #"round-robin"
  #"least-total-load"
  "least-waiting"
  #"least-kv-cache"
  "autellix"
  #"kv-cache"
  "kv-threshold-autellix"
) 
# RATES=("32")
# RATES=("50")
RATES=("4" "8" "16" "32" "64")

# ============================================================
# Helper: scrape per-engine metrics
# ============================================================

get_prefix_metrics_per_engine() {
    local url=$1

    curl -s "$url" | awk '

    /^vllm:prefix_cache_queries_total{/ {
        match($0, /engine="([^"]+)"/, m)
        match($0, /} ([0-9.e+-]+)/, v)
        queries[m[1]] = v[1]
    }

    /^vllm:prefix_cache_hits_total{/ {
        match($0, /engine="([^"]+)"/, m)
        match($0, /} ([0-9.e+-]+)/, v)
        hits[m[1]] = v[1]
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

for scheduler in "${SCHEDULERS[@]}"; do
for rate in "${RATES[@]}"; do

     # ------------------------------------
     # Start proxy server
     # ------------------------------------
     # variar o sheduler
    SCHEDULER="$scheduler" \
    LOAD_BALANCER_STRATEGY="round-robin" \
    python main.py &
    PROXY_PID=$!
    sleep 10  # tempo para proxy + engines estabilizarem
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
        # colocar na mesma pasta dos outros
        OUTPUT_JSON="${OUTPUTS_DIR}/baseline_output_${scheduler}_round-robin_rate${rate}_run${RUN}.json"
        python benchmark_stateful.py \
            --base-url "$BASE_URL" \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --limit "$LIMIT" \
            --request-rate "$rate" \
            --mode "chat" \
            --output-json "$OUTPUT_JSON" \
            --chat_len 25 \
            --is_baseline_run 1 \
            --burstiness 0.15
    done
    #Matando o proxy
    echo "Stopping proxy (PID=$PROXY_PID)"
    kill "$PROXY_PID"
    wait "$PROXY_PID" 2>/dev/null || true
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

#         # ------------------------------------
#         # Build JSON object
#         # ------------------------------------
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
        --arg scheduler "$scheduler" \
        --arg strategy "$strategy" \
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

for scheduler in "${SCHEDULERS[@]}"; do
for rate in "${RATES[@]}"; do

for strategy in "${LOAD_BALANCER_STRATEGIES[@]}"; do


    echo "======================================"
    echo "Starting proxy:"
    echo "  Scheduler = $scheduler"
    echo "  LB        = $strategy"
    echo "  Rate      = $rate"
    echo "======================================"

    # ------------------------------------
    # Start proxy server
    # ------------------------------------
    SCHEDULER="$scheduler" \
    LOAD_BALANCER_STRATEGY="$strategy" \
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
        OUTPUT_JSON="${OUTPUTS_DIR}/output_${scheduler}_${strategy}_rate${rate}_run${RUN}.json"
        python benchmark_stateful.py \
            --base-url "$BASE_URL" \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --limit "$LIMIT" \
            --request-rate "$rate" \
            --mode "chat" \
            --output-json "$OUTPUT_JSON" \
            --chat_len 25 \
            --is_baseline_run 0 \
            --burstiness 0.15

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
        --arg scheduler "$scheduler" \
        --arg strategy "$strategy" \
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
    sleep 5
done
done
done
mv /scratch/global/proxy_server/kv_cache_usage.csv "$OUTPUTS_DIR/kv_cache_usage.csv"

echo "======================================"
echo "All benchmarks completed successfully."
echo "======================================"
