import time
from threading import Lock
from config.settings import KV_CACHE_THRESHOLD, ROUTING_MODE, BACKEND_METRICS

metrics_lock = Lock()

# global metrics storage
backend_metrics = BACKEND_METRICS


def init_metrics(backends):
    global backend_metrics
    backend_metrics = {
        url: {"running": 0, "waiting": 0, "kv_cache": 0.0, "last_updated": 0}
        for url in backends
    }


def select_best_backend(candidates):
    now = time.time()
    with metrics_lock:
        fresh = [
            (url, backend_metrics[url])
            for url in candidates
        ]

        if not fresh:
            return None

        if ROUTING_MODE == "kv_limit":
            allowed = [
                (u, m)
                for u, m in fresh
                if m["kv_cache"] < KV_CACHE_THRESHOLD and m["waiting"] < 5
            ]
            if not allowed:
                return None
            return min(allowed, key=lambda x: x[1]["running"] + x[1]["waiting"])[0]

        if ROUTING_MODE == "min_waiting":
            return min(fresh, key=lambda x: x[1]["waiting"])[0]
