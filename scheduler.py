import re
import threading
import time
import csv
import asyncio
from datetime import datetime
from fastapi import FastAPI, Request
import httpx
import uvicorn
from threading import Lock
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse

MODEL_ROUTES = {
    "Qwen/Qwen3-4B": [
        # "http://localhost:8105",
        "http://localhost:8106",
        ],
}

# Collect all backend URLs from MODEL_ROUTES (flattened list)
ALL_BACKENDS = sorted({url for backends in MODEL_ROUTES.values() for url in backends})

# Initialize backend metrics and queues without relying on BACKEND_LOGS
BACKEND_METRICS = {
    url: {"running": 0, "waiting": 0, "kv_cache": 0.0, "last_updated": 0}
    for url in ALL_BACKENDS
}

BACKEND_QUEUE = {url: 0 for url in ALL_BACKENDS}

queue_lock = Lock()
metrics_lock = Lock()

metrics_lock = Lock()

# --- Métricas globais ---
LOG_FILE = "proxy_metrics.csv"
log_lock = Lock()
monitoring_active = False
first_request_time = None


import httpx
from prometheus_client.parser import text_string_to_metric_families
import asyncio

async def fetch_metrics_once(backend_url, timeout=1.5):
    """Try fetching /metrics once, with timeout."""
    metrics_url = f"{backend_url}/metrics"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            resp = await client.get(metrics_url)
            if resp.status_code == 200:
                return resp.text
    except Exception as e:
        print(f"[WARN] {backend_url} metrics timeout or error: {e}")
    return None


async def monitor_metrics(backend_url, interval=1.0):
    """Fetch vLLM metrics periodically and update BACKEND_METRICS."""
    while True:
        raw_metrics = await fetch_metrics_once(backend_url)
        if raw_metrics:
            try:
                metrics_data = {}
                for family in text_string_to_metric_families(raw_metrics):
                    metrics_data[family.name] = {
                        s.name: s.value for s in family.samples
                    }

                running = int(metrics_data.get("vllm:num_requests_running", {}).get("vllm:num_requests_running", 0))
                waiting = int(metrics_data.get("vllm:num_requests_waiting", {}).get("vllm:num_requests_waiting", 0))
                raw_kv = float(metrics_data.get("vllm:kv_cache_usage_perc", {}).get("vllm:kv_cache_usage_perc", 0.0))
                # Se o valor for entre 0–1, converte para porcentagem
                kv_cache = raw_kv * 100 if raw_kv <= 1 else raw_kv


                with metrics_lock:
                    BACKEND_METRICS[backend_url].update({
                        "running": running,
                        "waiting": waiting,
                        "kv_cache": kv_cache,
                        "last_updated": time.time()
                    })
            except Exception as e:
                print(f"[ERROR] Failed to parse metrics from {backend_url}: {e}")
        await asyncio.sleep(interval)


def select_best_backend(candidates):
    """Seleciona backend conforme a política definida."""
    now = time.time()
    with metrics_lock:
        valid = []
        for url in candidates:
            m = BACKEND_METRICS[url]
            if now - m["last_updated"] < STALE_TIMEOUT:
                valid.append((url, m))

        if not valid:
            print("[WARN] Nenhum backend com métricas atualizadas.")
            return None

        if ROUTING_MODE == "kv_limit":
            allowed = [(url, m) for url, m in valid if (m["kv_cache"] < KV_CACHE_THRESHOLD and m["waiting"] < 5)]
            if not allowed:
                print(f"[WAIT] Todos os backends com KV >= {KV_CACHE_THRESHOLD:.1f}% — aguardando...")
                return None
            best_url, _ = min(allowed, key=lambda item: item[1]["running"] + item[1]["waiting"])
            return best_url

        elif ROUTING_MODE == "min_waiting":
            best_url, _ = min(valid, key=lambda item: item[1]["waiting"])
            return best_url

def acquire_backend(url):
    with queue_lock:
        BACKEND_QUEUE[url] += 1
    return url

def release_backend(backend_url):
    """Decrement counter atomically."""
    with queue_lock:
        if BACKEND_QUEUE[backend_url] > 0:
            BACKEND_QUEUE[backend_url] -= 1
        # print(f"[QUEUE] Released {backend_url} (queue={BACKEND_QUEUE[backend_url]})")


def log_latency(latency, backend_url):
    """Salva a latência da requisição em CSV"""
    with log_lock:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), latency, backend_url])


def tail_log_file(path, pattern):
    """Yield new log lines matching pattern."""
    try:
        with open(path, "r") as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                if pattern.search(line):
                    yield line
    except FileNotFoundError:
        print(f"[WARN] Log file {path} not found")
        while True:
            time.sleep(5)



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Starting vLLM metrics monitoring tasks...")
    for backend_url in ALL_BACKENDS:
        asyncio.create_task(monitor_metrics(backend_url))
    yield
    print("[INFO] Shutting down proxy...")


app = FastAPI(lifespan=lifespan)


def print_backend_status():
    """Print formatted metrics for all backends."""
    with metrics_lock:
        print("\n[STATUS] Current backend load:")
        print("-" * 70)
        for url, metrics in BACKEND_METRICS.items():
            port = url.split(":")[-1]
            running = metrics["running"]
            waiting = metrics["waiting"]
            kv = metrics["kv_cache"]
            updated_ago = time.time() - metrics["last_updated"]
            print(
                f"  • Port {port}: running={running:2d}, waiting={waiting:2d}, "
                f"KV={kv:5.1f}%, updated {updated_ago:4.1f}s ago"
            )
        print("-" * 70 + "\n")

@app.get("/queue")
async def get_queue_state():
    with queue_lock:
        return BACKEND_QUEUE.copy()

@app.get("/status")
async def get_status():
    """Return current backend metrics for visualization."""
    with metrics_lock:
        return BACKEND_METRICS


@app.post("/{id}/v1/chat/completions")
async def proxy_chat(request: Request, id: str):
    print(f"[INFO] Received chat request from id: {id}")
    global monitoring_active, first_request_time

    start_time = time.time()
    body = await request.json()
    model = body.get("model")
    # temperature = body.get("temperature", "<not_set>")
    # print(f"[TEMP] Temperature da requisição ({model}): {temperature}")


    # Ativa o monitoramento somente quando chega a primeira requisição
    if not monitoring_active:
        monitoring_active = True
        first_request_time = start_time
        print(
            f"[METRICS] Monitoring started at {datetime.now().isoformat()} "
            "(first request received)"
        )

    if model not in MODEL_ROUTES:
        return {"error": f"Unknown model: {model}"}

    candidates = MODEL_ROUTES[model]

    with metrics_lock:
        current_metrics = {url: BACKEND_METRICS[url].copy() for url in candidates}

    print(f"\n[REQUEST] Received new request for model: {model}")
    print_backend_status()

    now = time.time()
    valid_backends = [
        url for url in candidates
        if now - current_metrics[url].get("last_updated", 0) < 30
    ]

    best_backend = None
    while best_backend is None:
        best_backend = select_best_backend(candidates)
        if best_backend is None:
            await asyncio.sleep(1.0)  # espera até um backend ficar disponível

    acquire_backend(best_backend)



    total_loads = {
        url: current_metrics[url]["running"] + current_metrics[url]["waiting"]
        for url in valid_backends
    }

    print(f"[INFO] Load summary: {total_loads}")
    print(f"[INFO] Selected backend for {model}: {best_backend}\n")

    target_url = f"{best_backend}/v1/chat/completions"
    
    try:
        async with httpx.AsyncClient(timeout=36000.0) as client:
            resp = await client.post(target_url, json=body)

        latency = time.time() - start_time
        log_latency(latency, best_backend)  # salva no CSV
        print(f"[METRIC] E2E latency = {latency:.3f}s | Backend = {best_backend}")

        return resp.json()
    except httpx.RequestError as e:
        latency = time.time() - start_time
        log_latency(latency, best_backend)
        print(f"[ERROR] Failed request ({latency:.3f}s): {e}")
        return {"error": f"Backend unavailable: {best_backend}"}
    
    finally:
        release_backend(best_backend)




@app.post("/{id}/v1/completions")
async def proxy_completion(request: Request, id: str):
    print(f"[INFO] Received completion request from id: {id}")
    global monitoring_active, first_request_time

    start_time = time.time()
    body = await request.json()
    model = body.get("model")
    # temperature = body.get("temperature", "<not_set>")
    # print(f"[TEMP] Temperature da requisição ({model}): {temperature}")


    if not monitoring_active:
        monitoring_active = True
        first_request_time = start_time
        print(f"[METRICS] Monitoring started at {datetime.now().isoformat()} (first request)")

    if model not in MODEL_ROUTES:
        return {"error": f"Unknown model: {model}"}

    candidates = MODEL_ROUTES[model]


    best_backend = acquire_backend(candidates)

    print(f"[INFO] Selected backend for {model}: {best_backend}")


    async def stream_generator():
        try:
            async with httpx.AsyncClient(timeout=36000.0) as client:
                async with client.stream("POST", f"{best_backend}/v1/completions", json=body) as backend_resp:
                    async for chunk in backend_resp.aiter_raw():
                        yield chunk
        finally:
            release_backend(best_backend)


    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",  # SSE from vLLM
        status_code=200,
    )



if __name__ == "__main__":
    # --- POLÍTICAS DE ROTEAMENTO ---
    ROUTING_MODE = "kv_limit"     # opções: "kv_limit" | "min_waiting"
    KV_CACHE_THRESHOLD = 90.0      # só envia se KV < 95%
    STALE_TIMEOUT = 30.0           # tempo limite em segundos para considerar métricas atualizadas

    uvicorn.run(app, host="0.0.0.0", port=8080)


#TODO:
# fazer o scheduler enviar as requisicoes de pouco em pouco para os servidores, se nao ele envia demais e sobrecarrega, no intervalo entre medicoes