# routes/chat.py

from fastapi import APIRouter, Request, HTTPException
import asyncio

from config.settings import MODEL_ROUTES, DISPATCH_MODE
from routing.selector import select_best_backend, backend_metrics
from logs.logger import print_backend_status
from uuid import uuid4
from routing.process_table import PROCESS_TABLE

# Fila unificada
from routing.queue_manager import init_queues
init_queues()

# Lazy initialization
dispatcher = None

router = APIRouter()


def get_dispatcher():
    global dispatcher

    # inicializa só 1 vez
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


@router.post("/{id}/v1/chat/completions")
async def chat_completion(id: str, request: Request):
    data = await request.json()
    model = data.get("model")

    candidates = MODEL_ROUTES.get(model)
    if not candidates:
        raise HTTPException(400, f"Unknown model: {model}")

    print_backend_status(backend_metrics)

    backend = None
    while backend is None:
        backend = select_best_backend(candidates)
        if backend is None:
            await asyncio.sleep(1)

    # instrument arrival (program id = path param `id`)
    call_id = str(uuid4())
    PROCESS_TABLE.record_call_arrival(id, call_id)

    disp = get_dispatcher()

    return await disp.dispatch(backend, data, program_id=id, call_id=call_id)
