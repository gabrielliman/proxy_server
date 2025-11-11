import asyncio
import httpx
from config.settings import ALL_BACKENDS, METRICS_INTERVAL, METRICS_TIMEOUT, BACKEND_METRICS
from metrics.parser import parse_prometheus_metrics
from routing.selector import backend_metrics, metrics_lock

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
        raw = await fetch_metrics_once(url)
        if raw:
            parsed = parse_prometheus_metrics(raw)
            with metrics_lock:
                backend_metrics[url].update(parsed)
                backend_metrics[url]["last_updated"] = asyncio.get_event_loop().time()

        await asyncio.sleep(METRICS_INTERVAL)


def start_monitoring_tasks(app):
    for backend_url in ALL_BACKENDS:
        asyncio.create_task(monitor_backend(backend_url))
