#!/bin/bash
set -e

# ============================================================
# Environment
# ============================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/global/abacus

BASE_URL="http://localhost:8080"
DATASET="/scratch/global/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
MODEL="Qwen/Qwen3-4B"
LIMIT=10
OUTPUTS_DIR="outputs_benchmark_load_balancer_2_engines"

mkdir -p "$OUTPUTS_DIR"

# ============================================================
# Engines (metrics + reset)
# OBS: benchmark coleta métricas apenas de UM endpoint
# ============================================================
RESET_URLS=(
  "http://localhost:8105/reset_prefix_cache"
  "http://localhost:8106/reset_prefix_cache"
)

# ============================================================
# Experiment grid
# ============================================================
REPEATS=1

SCHEDULERS=("fcfs" "plas") 
LOAD_BALANCER_STRATEGIES=(
  "round-robin"
#   "least-total-load"
  "least-waiting"
#   "least-kv-cache"
  "autellix"
#   "kv-cache"
  "kv-threshold-autellix"
) 
RATES=("0.25" "0.5" "0.75" "1.0" "1.25" "1.5" "2.0")
# RATES=("0.2" "0.4" "0.6" "0.8" "1.0" "1.3" "1.6" "2.0" "3.0" "5.0")

# ============================================================
# MAIN LOOP
# ============================================================

for scheduler in "${SCHEDULERS[@]}"; do
for strategy in "${LOAD_BALANCER_STRATEGIES[@]}"; do
for rate in "${RATES[@]}"; do

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

        python benchmark_stateful.py \
            --base-url "$BASE_URL" \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --limit "$LIMIT" \
            --request-rate "$rate" \
            --mode "chat" \
            --max_chat_len "25" \
            --output-json "${OUTPUTS_DIR}/output_${scheduler}_${strategy}_rate${rate}_run${RUN}.json"

        # -------------------------------
        # Reset prefix cache (all engines)
        # -------------------------------
        for url in "${RESET_URLS[@]}"; do
            curl -s -X POST "$url" > /dev/null
        done

        echo "Prefix cache reset done"
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

echo "======================================"
echo "All benchmarks completed successfully."
echo "======================================"
