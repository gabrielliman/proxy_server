# routing/dispatcher_direct.py

import httpx
import time
import asyncio

from config.settings import REQUEST_TIMEOUT
from routing.dispatcher.base import BaseDispatcher
from routing.queue_manager import get_queue, put_request, get_request
from config.settings import SCHEDULER
from routing.scheduler_plas import compute_priority


class DirectDispatcher(BaseDispatcher):
    """
    Implementação do modo DIRECT usando a fila real:
    - Enfileira
    - Remove imediatamente
    - Envia requisição via HTTP
    """

    async def dispatch(self, backend: str, data: dict, program_id: str = None, call_id: str = None) -> dict:
        queue = get_queue(backend)

        # Enfileira a requisição (permite contagem real da fila)
        future = asyncio.get_running_loop().create_future()
        from routing.process_table import PROCESS_TABLE
        if program_id and call_id:
            PROCESS_TABLE.record_enqueue(program_id, call_id)

        item = {"data": data, "future": future, "program_id": program_id, "call_id": call_id}
        if SCHEDULER == "plas":
            priority, seq = compute_priority(program_id, call_id)
            await put_request(backend, priority, item)
        else:
            await queue.put(item)

        # Remove imediatamente (modo DIRECT não espera)
        if SCHEDULER == "plas":
            item = await get_request(backend)
        else:
            item = await queue.get()
        pid = item.get("program_id")
        cid = item.get("call_id")
        # mark dequeue for waiting-time measurement
        if pid and cid:
            PROCESS_TABLE.record_dequeue(pid, cid)
        # call task_done if supported (asyncio.Queue)
        if hasattr(queue, "task_done") and not SCHEDULER == "plas":
            queue.task_done()

        data = item["data"]
        future = item["future"]
        pid = item.get("program_id")
        cid = item.get("call_id")

        # Instrumentation: record start/completion around the actual HTTP call
        from routing.process_table import PROCESS_TABLE

        start = None
        try:
            start = time.time()
            if pid and cid:
                PROCESS_TABLE.record_call_start(pid, cid, engine_id=backend, start_time=start)

            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
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
