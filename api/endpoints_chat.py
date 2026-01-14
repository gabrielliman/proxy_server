# routes/chat.py

from fastapi import APIRouter, Request, HTTPException
import asyncio
from uuid import uuid4

from config.settings import (
    MODEL_ROUTES,
    DISPATCH_MODE,
)

from routing.load_balancer import LOAD_BALANCER
from routing.process_table import PROCESS_TABLE
from logs.logger import print_backend_status

# Unified queue initialization (run once)
from routing.queue_manager import init_queues
init_queues()

# Lazy dispatcher initialization
dispatcher = None

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
    # Estimate input tokens (cheap heuristic)
    # -----------------------------
    messages = data.get("messages", [])
    num_input_tokens = sum(
        len(m.get("content", "")) for m in messages if isinstance(m, dict)
    )

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
    # Dispatch
    # -----------------------------
    disp = get_dispatcher()

    return await disp.dispatch(
        backend=backend,
        data=data,
        program_id=program_id,
        call_id=call_id,
    )
