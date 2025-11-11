import csv
from datetime import datetime
from config.settings import LOG_FILE
from threading import Lock

log_lock = Lock()

def log_latency(latency, backend):
    with log_lock:
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(), latency, backend])


def print_backend_status(metrics):
    print("\n[STATUS] Current backend load:")
    print("-" * 70)
    for url, m in metrics.items():
        print(
            f"{url}: running={m['running']}, waiting={m['waiting']}, "
            f"kv={m['kv_cache']:.1f}%, updated={m['last_updated']}"
        )
    print("-" * 70)
