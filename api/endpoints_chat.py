# routes/chat.py

from fastapi import APIRouter, Request, HTTPException
import asyncio
from uuid import uuid4

from config.settings import (
    MODEL_ROUTES,
    DISPATCH_MODE,
    MODEL,
)

from routing.load_balancer import LOAD_BALANCER
from routing.process_table import PROCESS_TABLE
from logs.logger import print_backend_status

# Unified queue initialization (run once)
from routing.queue_manager import init_queues
init_queues()

# Lazy dispatcher initialization
dispatcher = None

from utils.tokenizer_utils import get_tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in text using tokenizer."""
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))

router = APIRouter()


def get_dispatcher():
    global dispatcher

    if dispatcher is not None:
        return dispatcher

    if DISPATCH_MODE == "direct":
        from routing.dispatcher.direct import DirectDispatcher
        dispatcher = DirectDispatcher()

    elif DISPATCH_MODE == "worker_pool":
        from routing.dispatcher.worker import WorkerPoolDispatcher
        dispatcher = WorkerPoolDispatcher()

    elif DISPATCH_MODE == "semaphore":
        from routing.dispatcher.semaphore import SemaphoreDispatcher
        dispatcher = SemaphoreDispatcher()

    else:
        raise RuntimeError(f"Invalid DISPATCH_MODE: {DISPATCH_MODE}")

    return dispatcher


@router.post("/{program_id}/v1/chat/completions")
async def chat_completion(program_id: str, request: Request):
    data = await request.json()
    model = data.get("model")

    if not model:
        raise HTTPException(400, "Missing model field")

    candidates = MODEL_ROUTES.get(model)
    if not candidates:
        raise HTTPException(400, f"Unknown model: {model}")

    # -----------------------------
    # Estimate input tokens (using tokenizer for accurate count)
    # -----------------------------
    messages = data.get("messages", [])
    # Combine all message content for token counting
    input_text = "\n".join(m.get("content", "") for m in messages if isinstance(m, dict))
    num_input_tokens = count_tokens(input_text)


    # -----------------------------
    # Instrument arrival (with prefill tokens for KV metric)
    # -----------------------------
    call_id = str(uuid4())
    PROCESS_TABLE.record_call_arrival(program_id, call_id, prefill_tokens=num_input_tokens)

    # -----------------------------
    # Dispatch
    # -----------------------------
    disp = get_dispatcher()

    return await disp.dispatch(
        backend="global_cluster",
        data=data,
        program_id=program_id,
        call_id=call_id,
        input_tokens=num_input_tokens
    )

