from fastapi import APIRouter
from routing.selector import backend_metrics, metrics_lock
from request_queue.manager import get_queue_state
import asyncio
from routing.process_table import PROCESS_TABLE

router = APIRouter()


@router.get("/status")
async def status():
    with metrics_lock:
        return backend_metrics


@router.get("/queue")
async def queue_state():
    return get_queue_state()


@router.get("/processes")
async def processes():
    # return the current process table snapshot
    print("Fetching process table snapshot")
    return PROCESS_TABLE.list_processes()
