from fastapi import APIRouter, Request
import httpx
import time
import asyncio

from config.settings import MODEL_ROUTES, REQUEST_TIMEOUT
from routing.selector import select_best_backend, backend_metrics, metrics_lock
from request_queue.manager import acquire_backend, release_backend
from logs.logger import log_latency, print_backend_status

router = APIRouter()


@router.post("/{id}/v1/chat/completions")
async def chat_completion(id: str, request: Request):
    data = await request.json()
    model = data.get("model")

    candidates = MODEL_ROUTES.get(model)
    if not candidates:
        return {"error": f"Unknown model: {model}"}

    print_backend_status(backend_metrics)

    backend = None
    while backend is None:
        backend = select_best_backend(candidates)
        if backend is None:
            await asyncio.sleep(1)

    acquire_backend(backend)

    target_url = f"{backend}/v1/chat/completions"

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(target_url, json=data)
        latency = time.time() - start
        log_latency(latency, backend)
        return resp.json()
    finally:
        release_backend(backend)
