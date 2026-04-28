import os
#CONFIGURE AQUI
MODEL= "meta-llama/Llama-3.1-8B-Instruct"
MODEL_ROUTES = {
    # "Qwen/Qwen3-4B": [
    #     "http://localhost:8105",
    #     # "http://localhost:8105"
    # ],
    
    "meta-llama/Llama-3.1-8B-Instruct": [
        "http://localhost:8105",
        # "http://localhost:8106"
    ],
}

#nao funciona para o endpoint completions
BACKEND_PARALLELISM = {
    "http://localhost:8105": 15, #50% a mais que o paralelismo real
    "http://localhost:8106": 15,
}
#FIM
ALL_BACKENDS = sorted({url for lst in MODEL_ROUTES.values() for url in lst})

BACKEND_METRICS = {
    url: {"running": 0, "waiting": 0, "kv_cache": 0.0, "last_updated": 0}
    for url in ALL_BACKENDS
}


LOG_FILE = "proxy_metrics.csv"
METRICS_INTERVAL = 1.0
REQUEST_TIMEOUT = 36000.0
METRICS_TIMEOUT = 1.5


DISPATCH_MODE = "worker_pool"  # "direct", "worker_pool", "semaphore"

 
# Process table pruning (seconds)
PROCESS_TABLE_PRUNE_TTL = 600  # default 10 minutes
PROCESS_TABLE_PRUNE_INTERVAL = 60  # run pruner every 60 seconds

# Scheduler selection: 'fcfs' (default), 'plas', ...
SCHEDULER = os.getenv("SCHEDULER", "plas")
PLAS_EWMA_ALPHA = 0.3
PLAS_AGE_WEIGHT = 0.5
PLAS_PRIORITY_BASE = 1.0

# PLAS metric type: 'service_ewma' (default, LAS) or 'kv_token_time'
# - service_ewma: Uses EWMA of actual service time (Least Attained Service)
# - kv_token_time: Uses EWMA of KV token-time (d*c = pd + d²/2)
PLAS_METRIC_TYPE = os.getenv("PLAS_METRIC_TYPE", "kv_token_time")

# Discretized priority queues for PLAS (Autellix-style)
DISCRETIZED_PRIORITY_BUCKETS = 10  # K: number of priority levels
DISCRETIZED_PRIORITY_BASE = PLAS_PRIORITY_BASE * 2  # span to discretize (adjust as needed)
ANTI_STARVATION_RATIO_THRESHOLD = 0.5  # beta: W_total/T_total >= threshold triggers promotion to Q1

LOAD_BALANCER_SHORT_REQUEST_THRESHOLD = 2048
# Load balancer configuration
LOAD_BALANCER_ENABLE = True
LOAD_BALANCER_STRATEGY = os.getenv("LOAD_BALANCER_STRATEGY", "autellix")  # options: "autellix", "kv-cache"
LOAD_BALANCER_METRICS_LOG_INTERVAL = 5  # seconds between human-readable logs
LOAD_BALANCER_ENABLE_METRICS_QUERY = True  # query engine /metrics endpoints