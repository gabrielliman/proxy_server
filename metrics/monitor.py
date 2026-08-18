import asyncio
import httpx
from config.settings import ALL_BACKENDS, METRICS_INTERVAL, METRICS_TIMEOUT, BACKEND_METRICS
from metrics.parser import parse_prometheus_metrics
from routing.selector import backend_metrics, metrics_lock
from metrics.csv_writer import log_kv_cache
from routing.process_table import PROCESS_TABLE

backend_metrics = BACKEND_METRICS

async def fetch_metrics_once(url):
    try:
        async with httpx.AsyncClient(timeout=METRICS_TIMEOUT) as client:
            r = await client.get(f"{url}/metrics")
            return r.text if r.status_code == 200 else None
    except Exception:
        return None


async def monitor_backend(url):
    while True:
        try:
            raw = await fetch_metrics_once(url)

            if raw:
                parsed = parse_prometheus_metrics(raw)

                incomplete_count = PROCESS_TABLE.get_incomplete_requests_count()
                waiting_count = PROCESS_TABLE.get_waiting_requests_count()

                log_kv_cache(url, parsed.get("kv_cache", []), incomplete_count, waiting_count)

                with metrics_lock:
                    backend_metrics[url].update(parsed)
                    backend_metrics[url]["last_updated"] = asyncio.get_event_loop().time()
                    # You can safely keep it here if your /status endpoint needs it
                    backend_metrics[url]["incomplete_count"] = incomplete_count
                    backend_metrics[url]["proxy_waiting_count"] = waiting_count

        except Exception:
            import traceback
            traceback.print_exc()

        await asyncio.sleep(METRICS_INTERVAL)


def start_monitoring_tasks(app):
    # Only the backend monitors are needed now!
    for backend_url in ALL_BACKENDS:
        asyncio.create_task(monitor_backend(backend_url))