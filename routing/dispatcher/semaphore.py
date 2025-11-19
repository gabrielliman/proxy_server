# routing/dispatcher/semaphore.py

import asyncio
import httpx
from routing.dispatcher.base import BaseDispatcher
from routing.queue_manager import get_queue
from config.settings import BACKEND_PARALLELISM, REQUEST_TIMEOUT


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

    async def dispatch(self, backend: str, data: dict) -> dict:
        await self.start_backend(backend)

        queue = get_queue(backend)
        loop = asyncio.get_running_loop()

        # Enfileira para contagem
        future = loop.create_future()
        await queue.put({"data": data, "future": future})

        # Remove imediatamente (como DIRECT)
        item = await queue.get()
        queue.task_done()

        data = item["data"]
        future = item["future"]

        sem = self.semaphores[backend]
        client = self.clients[backend]

        # 🔥 Entrada no semáforo = limite de concorrência real
        async with sem:
            try:
                resp = await client.post(f"{backend}/v1/chat/completions", json=data)
                if not future.done():
                    future.set_result(resp.json())
            except Exception as e:
                if not future.done():
                    future.set_exception(e)

        return await future
