from fastapi import APIRouter
from routing.selector import backend_metrics, metrics_lock
from request_queue.manager import get_queue_state
import asyncio

router = APIRouter()


@router.get("/status")
async def status():
    with metrics_lock:
        return backend_metrics


@router.get("/queue")
async def queue_state():
    return get_queue_state()
