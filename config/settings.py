MODEL_ROUTES = {
    "Qwen/Qwen3-4B": [
        "http://localhost:8106"
    ],
}

ALL_BACKENDS = sorted({url for lst in MODEL_ROUTES.values() for url in lst})

BACKEND_METRICS = {
    url: {"running": 0, "waiting": 0, "kv_cache": 0.0, "last_updated": 0}
    for url in ALL_BACKENDS
}
ROUTING_MODE = "min_waiting"
KV_CACHE_THRESHOLD = 90.0
#STALE_TIMEOUT = 30.0

LOG_FILE = "proxy_metrics.csv"
METRICS_INTERVAL = 1.0
REQUEST_TIMEOUT = 36000.0
METRICS_TIMEOUT = 1.5


DISPATCH_MODE = "worker_pool"  # "direct", "worker_pool", "semaphore"

BACKEND_PARALLELISM = {
    "http://localhost:8106": 10,
}