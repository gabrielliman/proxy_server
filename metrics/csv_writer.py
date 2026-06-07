import csv
import os
import time

CSV_PATH = "kv_cache_usage.csv"

HEADERS = [
    "timestamp",
    "backend",
    "engine",
    "model_name",
    "kv_cache_percent",
    "running",
    "waiting"
]


def log_kv_cache(backend_url, kv_list):
    if not kv_list:
        return

    file_exists = os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(HEADERS)

        now = time.time()

        for kv in kv_list:
            writer.writerow([
                now,
                backend_url,
                kv["engine"],
                kv["model_name"],
                kv["kv_cache"],
                kv.get("running", 0),
                kv.get("waiting", 0)
            ])
