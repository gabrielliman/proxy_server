# routing/dispatcher/workers.py

import asyncio
import httpx
import time
from routing.dispatcher.base import BaseDispatcher
from routing.queue_manager import get_queue
from config.settings import REQUEST_TIMEOUT, BACKEND_PARALLELISM, ALL_BACKENDS


class WorkerPoolDispatcher(BaseDispatcher):
    workers_started = False

    async def start_workers(self):
        if WorkerPoolDispatcher.workers_started:
            return
        WorkerPoolDispatcher.workers_started = True

        loop = asyncio.get_running_loop()

        for backend in ALL_BACKENDS:
            queue = get_queue(backend)
            n_workers = BACKEND_PARALLELISM.get(backend, 1)

            for worker_id in range(n_workers):
                loop.create_task(self.worker_loop(backend, queue, worker_id))

    async def worker_loop(self, backend, queue, worker_id):

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            while True:
                item = await queue.get()
                data = item["data"]
                future = item["future"]

                try:
                    resp = await client.post(f"{backend}/v1/chat/completions", json=data)

                    if not future.done():
                        future.set_result(resp.json())

                except Exception as e:
                    print(f"[WORKER ERROR] {e}")
                    if not future.done():
                        future.set_exception(e)

                finally:
                    queue.task_done()

    async def dispatch(self, backend: str, data: dict) -> dict:
        await self.start_workers()

        queue = get_queue(backend)
        loop = asyncio.get_running_loop()

        future = loop.create_future()
        await queue.put({"data": data, "future": future})

        return await future
