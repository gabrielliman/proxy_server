# routing/dispatcher/workers.py

import asyncio
import httpx
import time
from routing.dispatcher.base import BaseDispatcher
from routing.queue_manager import get_queue, put_request, get_request
from config.settings import REQUEST_TIMEOUT, BACKEND_PARALLELISM, ALL_BACKENDS, SCHEDULER
from routing.scheduler_plas import compute_priority


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
                if SCHEDULER == "plas":
                    item = await get_request(backend)
                else:
                    item = await queue.get()
                data = item["data"]
                future = item["future"]
                pid = item.get("program_id")
                cid = item.get("call_id")

                from routing.process_table import PROCESS_TABLE

                # record dequeue event
                if pid and cid:
                    PROCESS_TABLE.record_dequeue(pid, cid)

                start = None
                try:
                    # mark start (waiting -> running)
                    start = time.time()
                    if pid and cid:
                        PROCESS_TABLE.record_call_start(pid, cid, engine_id=backend, start_time=start)

                    resp = await client.post(f"{backend}/v1/chat/completions", json=data)

                    if not future.done():
                        future.set_result(resp.json())

                except Exception as e:
                    print(f"[WORKER ERROR] {e}")
                    if not future.done():
                        future.set_exception(e)

                finally:
                    # completion
                    end = time.time()
                    if pid and cid:
                        PROCESS_TABLE.record_call_completion(pid, cid, completion_time=end)
                    # call task_done if supported (asyncio.Queue)
                    if hasattr(queue, "task_done") and not SCHEDULER == "plas":
                        queue.task_done()

    async def dispatch(self, backend: str, data: dict, program_id: str = None, call_id: str = None) -> dict:
        from routing.process_table import PROCESS_TABLE
        await self.start_workers()

        queue = get_queue(backend)
        loop = asyncio.get_running_loop()

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

        return await future

