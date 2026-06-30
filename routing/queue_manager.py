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


class DiscretizedQueueWrapper:
    """K discrete priority queues (Q1, Q2, ..., QK) with FCFS within each.
    
    Q1 = highest priority (index 0), QK = lowest priority (index K-1).
    Each queue is an asyncio.Queue; dequeue always selects from highest non-empty queue.
    """
    def __init__(self, num_buckets: int):
        self.num_buckets = num_buckets
        self.queues = [asyncio.Queue() for _ in range(num_buckets)]

    async def put(self, queue_index: int, item: Any):
        """Enqueue item to specified queue index (0 = Q1, highest priority)."""
        idx = min(max(queue_index, 0), self.num_buckets - 1)
        await self.queues[idx].put(item)

    async def get(self) -> Any:
        """Dequeue from highest non-empty queue (Q1, Q2, ...)."""
        while True:
            for qi in range(self.num_buckets):
                if not self.queues[qi].empty():
                    return await self.queues[qi].get()
            # All queues empty, wait a bit and retry
            await asyncio.sleep(0.001)

    def qsize(self) -> int:
        return sum(q.qsize() for q in self.queues)

    def empty(self) -> bool:
        return all(q.empty() for q in self.queues)


def init_queues():
    """Inicializa a fila global do cluster.
    Chamada automática na primeira importação do dispatcher.
    """
    global backend_queues

    if backend_queues:
        return

    # Usamos uma chave única para a fila global
    global_queue_name = "global_cluster"

    if SCHEDULER in ("plas", "atlas"):
        from config.settings import DISCRETIZED_PRIORITY_BUCKETS
        backend_queues = {
            global_queue_name: DiscretizedQueueWrapper(DISCRETIZED_PRIORITY_BUCKETS)
        }
    else:
        backend_queues = {global_queue_name: asyncio.Queue()}


def get_queue(backend: str):
    """Retorna a fila do backend (já inicializada)."""
    return backend_queues[backend]


async def put_request(backend: str, priority: float, item: Any):
    q = get_queue(backend)
    
    if SCHEDULER in ("plas", "atlas") and isinstance(q, DiscretizedQueueWrapper):
        if SCHEDULER == "atlas":
            from routing.scheduler_atlas import map_priority_to_queue_index
        else:
            from routing.scheduler_plas import map_priority_to_queue_index
            
        from config.settings import DISCRETIZED_PRIORITY_BUCKETS, DISCRETIZED_PRIORITY_BASE
        queue_idx = map_priority_to_queue_index(priority, DISCRETIZED_PRIORITY_BUCKETS, DISCRETIZED_PRIORITY_BASE)
        await q.put(queue_idx, item)
    elif SCHEDULER in ("plas", "atlas"):
        await q.put(priority, item)
    else:
        await q.put(item)

async def get_request(backend: str):
    q = get_queue(backend)
    if SCHEDULER in ("plas", "atlas") and isinstance(q, DiscretizedQueueWrapper):
        return await q.get()
    else:
        return await q.get()
