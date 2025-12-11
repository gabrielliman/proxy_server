# routing/dispatcher/semaphore.py

import asyncio
import httpx
import time
from routing.dispatcher.base import BaseDispatcher
from routing.queue_manager import get_queue, put_request, get_request
from config.settings import BACKEND_PARALLELISM, REQUEST_TIMEOUT, SCHEDULER
from routing.scheduler_plas import compute_priority


class SemaphoreDispatcher(BaseDispatcher):
    def __init__(self):
        # um semáforo por backend
        self.semaphores = {}
        self.clients = {}

    async def start_backend(self, backend):
        if backend not in self.semaphores:
            limit = BACKEND_PARALLELISM.get(backend, 1)
            self.semaphores[backend] = asyncio.Semaphore(limit)
            self.clients[backend] = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

    async def dispatch(self, backend: str, data: dict, program_id: str = None, call_id: str = None) -> dict:
        from routing.process_table import PROCESS_TABLE
        await self.start_backend(backend)

        queue = get_queue(backend)
        loop = asyncio.get_running_loop()

        # Enfileira para contagem
        future = loop.create_future()
        if program_id and call_id:
            PROCESS_TABLE.record_enqueue(program_id, call_id)

        item = {"data": data, "future": future, "program_id": program_id, "call_id": call_id}
        if SCHEDULER == "plas":
            from config.settings import ANTI_STARVATION_RATIO_THRESHOLD
            if program_id and PROCESS_TABLE.should_promote_to_q1(program_id, ANTI_STARVATION_RATIO_THRESHOLD):
                PROCESS_TABLE.reset_wait_and_service_for_promotion(program_id)
                await put_request(backend, 0.0, item)
            else:
                priority, seq = compute_priority(program_id, call_id)
                await put_request(backend, priority, item)
        else:
            await queue.put(item)

        # Remove immediately (como DIRECT)
        if SCHEDULER == "plas":
            item = await get_request(backend)
        else:
            item = await queue.get()
            queue.task_done()

        data = item["data"]
        future = item["future"]
        pid = item.get("program_id")
        cid = item.get("call_id")

        # record dequeue to mark leaving the queue
        if pid and cid:
            PROCESS_TABLE.record_dequeue(pid, cid)

        sem = self.semaphores[backend]
        client = self.clients[backend]

        # 🔥 Entrada no semáforo = limite de concorrência real
        async with sem:
            start = None
            try:
                start = time.time()
                if pid and cid:
                    PROCESS_TABLE.record_call_start(pid, cid, engine_id=backend, start_time=start)

                resp = await client.post(f"{backend}/v1/chat/completions", json=data)
                if not future.done():
                    future.set_result(resp.json())
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
                raise
            finally:
                end = time.time()
                if pid and cid:
                    PROCESS_TABLE.record_call_completion(pid, cid, completion_time=end)

        return await future
