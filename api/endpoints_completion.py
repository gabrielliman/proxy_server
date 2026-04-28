# routes/completions.py
#ACHO QUE NAO TA FUNCIONANDO COM O KVTOKENCACHE, NAO SEI SE MEDE OUTPUT TOKEN
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import asyncio
from uuid import uuid4

from config.settings import MODEL_ROUTES, REQUEST_TIMEOUT
from routing.load_balancer import LOAD_BALANCER
from routing.process_table import PROCESS_TABLE
from request_queue.manager import acquire_backend, release_backend

from utils.tokenizer_utils import get_tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in text using tokenizer."""
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))

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
    num_input_tokens = count_tokens(prompt)

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
    PROCESS_TABLE.record_call_arrival(program_id, call_id, prefill_tokens=num_input_tokens)

    # -----------------------------
    # Backend concurrency control
    # -----------------------------
    acquire_backend(backend)

    async def generator():
        output_text = ""
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
                        output_text += chunk.decode() if isinstance(chunk, bytes) else chunk
                        yield chunk

        finally:
            # Extract output tokens from accumulated response
            output_tokens = None
            if output_text:
                try:
                    import json
                    for line in output_text.strip().split('\n'):
                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove 'data: ' prefix
                            if data_str.strip() == '[DONE]':
                                continue
                            data = json.loads(data_str)
                            if "choices" in data:
                                choice = data["choices"][0] if data["choices"] else {}
                                text = choice.get("text", "") or choice.get("delta", {}).get("content", "")
                                if text:
                                    tok = get_tokenizer()
                                    output_tokens = len(tok.encode(text, add_special_tokens=False))
                                    break  # Get first valid text
                except Exception:
                    pass  # Ignore parsing errors
            
            # record completion with output tokens
            PROCESS_TABLE.record_call_completion(program_id, call_id, output_tokens=output_tokens)

            # release backend slot
            release_backend(backend)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
    )
