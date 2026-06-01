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
    prompt = body.get("prompt", "")
    num_input_tokens = count_tokens(prompt)
    backend = await LOAD_BALANCER.select_engine(
        program_id=program_id,
        num_input_tokens=num_input_tokens,
        candidates=candidates 
    )
    
    if backend is None:
        raise HTTPException(503, "No available backends for this model")

    call_id = str(uuid4())
    PROCESS_TABLE.record_call_arrival(program_id, call_id, prefill_tokens=num_input_tokens)

    acquire_backend(backend)

    async def generator():
        output_text = ""
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                PROCESS_TABLE.record_call_start(program_id, call_id, engine_id=backend)

                async with client.stream("POST", f"{backend}/v1/completions", json=body) as resp:
                    async for chunk in resp.aiter_raw():
                        # --- LÓGICA TIE: Poda durante o streaming ---
                        if await request.is_disconnected():
                            PROCESS_TABLE.record_call_error(program_id, call_id, "TIE_PRUNED")
                            print(f"✂️ [ENDPOINT COMPLETION] Streaming abortado. Liberando GPU: {backend}")
                            # Ao sair do loop e fechar o contexto do 'client', a conexão com o vLLM cai
                            return 

                        yield chunk

        finally:
            PROCESS_TABLE.record_call_completion(program_id, call_id)
            release_backend(backend)

    return StreamingResponse(generator(), media_type="text/event-stream")