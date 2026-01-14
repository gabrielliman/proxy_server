# routes/completions.py

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import asyncio
from uuid import uuid4

from config.settings import MODEL_ROUTES, REQUEST_TIMEOUT
from routing.load_balancer import LOAD_BALANCER
from routing.process_table import PROCESS_TABLE
from request_queue.manager import acquire_backend, release_backend

router = APIRouter()


@router.post("/{program_id}/v1/completions")
async def completions(program_id: str, request: Request):
    body = await request.json()
    model = body.get("model")

    if not model:
        raise HTTPException(400, "Missing model field")

    candidates = MODEL_ROUTES.get(model)
    if not candidates:
        raise HTTPException(400, f"Unknown model: {model}")

    # -----------------------------
    # Estimate input tokens (cheap heuristic)
    # -----------------------------
    prompt = body.get("prompt", "")
    num_input_tokens = len(prompt) if isinstance(prompt, str) else 0

    # -----------------------------
    # Select engine via Autellix LB
    # -----------------------------
    backend = None
    while backend is None:
        backend = await LOAD_BALANCER.select_engine(
            program_id=program_id,
            num_input_tokens=num_input_tokens,
        )
        if backend is None:
            await asyncio.sleep(0.05)

    # -----------------------------
    # Instrument arrival
    # -----------------------------
    call_id = str(uuid4())
    PROCESS_TABLE.record_call_arrival(program_id, call_id)

    # -----------------------------
    # Backend concurrency control
    # -----------------------------
    acquire_backend(backend)

    async def generator():
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                # record execution start (pins preferred_engine if needed)
                PROCESS_TABLE.record_call_start(
                    program_id,
                    call_id,
                    engine_id=backend,
                )

                async with client.stream(
                    "POST",
                    f"{backend}/v1/completions",
                    json=body,
                ) as resp:
                    async for chunk in resp.aiter_raw():
                        yield chunk

        finally:
            # record completion
            PROCESS_TABLE.record_call_completion(program_id, call_id)

            # release backend slot
            release_backend(backend)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
    )
