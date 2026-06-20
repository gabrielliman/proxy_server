import json
import os
from dotenv import load_dotenv

load_dotenv()

# CONFIGURE AQUI
HF_TOKEN = os.getenv("HF_TOKEN")
# 1. Simple string fallback (just like PLAS_METRIC_TYPE)
MODEL = os.getenv("MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# 2. Dictionary fallback for MODEL_ROUTES
_default_model_routes = {
    "meta-llama/Llama-3.1-8B-Instruct": [
        "http://localhost:8105",
        "http://localhost:8106"
    ],
}
_env_model_routes = os.getenv("MODEL_ROUTES")
# If the env var exists, parse it from JSON. Otherwise, use the default dict.
MODEL_ROUTES = json.loads(_env_model_routes) if _env_model_routes else _default_model_routes

# 3. Dictionary fallback for BACKEND_PARALLELISM
# nao funciona para o endpoint completions
_default_backend_parallelism = {
    "http://localhost:8105": 10,
    "http://localhost:8106": 10,
}
_env_backend_parallelism = os.getenv("BACKEND_PARALLELISM")
BACKEND_PARALLELISM = json.loads(_env_backend_parallelism) if _env_backend_parallelism else _default_backend_parallelism

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

# Scheduler selection: 'fcfs' (default), 'plas', "atlas"
SCHEDULER = os.getenv("SCHEDULER", "plas")

# PLAS metric type: 'service_cumulative' (default, LAS) or 'kv_token_time'
# - service_cumulative: Uses sum of actual service time (Least Attained Service)
# - kv_token_time: Uses sum of KV token-time (d*c = pd + d^2/2)
PLAS_METRIC_TYPE = os.getenv("PLAS_METRIC_TYPE", "service_cumulative")

# Discretized priority queues for PLAS (Autellix-style)
DISCRETIZED_PRIORITY_BUCKETS = 10  # K: number of priority levels
DISCRETIZED_PRIORITY_BASE = 50.0   # Adjust this span based on your expected max cumulative metric
ANTI_STARVATION_RATIO_THRESHOLD = 0.5  # beta: W_total/T_total >= threshold triggers promotion to Q1

LOAD_BALANCER_SHORT_REQUEST_THRESHOLD = int(os.getenv("LOAD_BALANCER_SHORT_REQUEST_THRESHOLD", 2048))

# Load balancer configuration
LOAD_BALANCER_ENABLE = True
LOAD_BALANCER_METRICS_LOG_INTERVAL = 5  # seconds between human-readable logs
LOAD_BALANCER_ENABLE_METRICS_QUERY = True  # query engine /metrics endpoints

# Options: "round-robin", "least-total-load", "least-waiting", "least-running", "least-kv-cache", "autellix", "threshold-autellix"
LOAD_BALANCER_STRATEGY = os.getenv("LOAD_BALANCER_STRATEGY", "autellix")

# Threshold Autellix Configuration
LOAD_BALANCER_THRESHOLD_METRIC = os.getenv("LOAD_BALANCER_THRESHOLD_METRIC", "kv_cache_percent")  # e.g., "kv_cache_percent", "waiting", "total", "running"
LOAD_BALANCER_THRESHOLD_VALUE = float(os.getenv("LOAD_BALANCER_THRESHOLD_VALUE", 90.0))           # e.g., 90.0, 10.0, etc.

# Options: "round-robin", "least-total-load", "least-waiting", "least-running", "least-kv-cache"
LOAD_BALANCER_FALLBACK_STRATEGY = os.getenv("LOAD_BALANCER_FALLBACK_STRATEGY", "least-total-load")