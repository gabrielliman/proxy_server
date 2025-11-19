import asyncio
from typing import Dict, Any
import itertools

from config.settings import ALL_BACKENDS, SCHEDULER


# Backend queues: either asyncio.Queue or PriorityQueueWrapper
backend_queues: Dict[str, Any] = {}


class PriorityQueueWrapper:
    """A small wrapper around asyncio.PriorityQueue that preserves FIFO for ties.

    Items are stored as (priority, seq, payload).
    """
    def __init__(self):
        self._pq = asyncio.PriorityQueue()
        self._seq = itertools.count()

    async def put(self, priority: float, item: Any):
        seq = next(self._seq)
        await self._pq.put((priority, seq, item))

    async def get(self):
        priority, seq, item = await self._pq.get()
        return item

    def qsize(self):
        return self._pq.qsize()

    def empty(self):
        return self._pq.empty()


def init_queues():
    """Inicializa filas para todos os backends.
    Chamada automática na primeira importação do dispatcher.
    """
    global backend_queues

    if backend_queues:
        return

    if SCHEDULER == "plas":
        backend_queues = {backend: PriorityQueueWrapper() for backend in ALL_BACKENDS}
    else:
        backend_queues = {backend: asyncio.Queue() for backend in ALL_BACKENDS}


def get_queue(backend: str):
    """Retorna a fila do backend (já inicializada)."""
    return backend_queues[backend]


async def put_request(backend: str, priority: float, item: Any):
    q = get_queue(backend)
    # PriorityQueueWrapper exposes put(priority, item); asyncio.Queue expects single item
    if SCHEDULER == "plas" and hasattr(q, "put"):
        await q.put(priority, item)
    else:
        # fallback: ignore priority and enqueue item
        await q.put(item)


async def get_request(backend: str):
    q = get_queue(backend)
    if SCHEDULER == "plas" and hasattr(q, "get") and isinstance(q, PriorityQueueWrapper):
        return await q.get()
    else:
        return await q.get()
