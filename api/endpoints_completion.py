from fastapi import APIRouter, Request
import httpx
import time
import asyncio

from config.settings import MODEL_ROUTES, REQUEST_TIMEOUT
from routing.selector import select_best_backend
from request_queue.manager import acquire_backend, release_backend

from fastapi.responses import StreamingResponse

router = APIRouter()


@router.post("/{id}/v1/completions")
async def completions(id: str, request: Request):
    body = await request.json()
    model = body.get("model")

    candidates = MODEL_ROUTES.get(model)
    if not candidates:
        return {"error": f"Unknown model: {model}"}

    backend = select_best_backend(candidates)
    if backend is None:
        return {"error": "No backend available"}

    acquire_backend(backend)

    async def generator():
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                async with client.stream("POST", f"{backend}/v1/completions", json=body) as r:
                    async for chunk in r.aiter_raw():
                        yield chunk
        finally:
            release_backend(backend)

    return StreamingResponse(generator(), media_type="text/event-stream")
