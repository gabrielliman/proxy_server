import time
from threading import Lock
from config.settings import BACKEND_METRICS

metrics_lock = Lock()

# global metrics storage
backend_metrics = BACKEND_METRICS


def init_metrics(backends):
    global backend_metrics
    backend_metrics = {
        url: {"running": 0, "waiting": 0, "kv_cache": 0.0, "last_updated": 0}
        for url in backends
    }