# routing/dispatcher_direct.py

import httpx
import time
import asyncio

from config.settings import REQUEST_TIMEOUT
from routing.dispatcher.base import BaseDispatcher
from routing.queue_manager import get_queue


class DirectDispatcher(BaseDispatcher):
    """
    Implementação do modo DIRECT usando a fila real:
    - Enfileira
    - Remove imediatamente
    - Envia requisição via HTTP
    """

    async def dispatch(self, backend: str, data: dict) -> dict:
        queue = get_queue(backend)

        # Enfileira a requisição (permite contagem real da fila)
        future = asyncio.get_running_loop().create_future()
        await queue.put({"data": data, "future": future})

        # Remove imediatamente (modo DIRECT não espera)
        item = await queue.get()
        queue.task_done()

        data = item["data"]
        future = item["future"]

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post(f"{backend}/v1/chat/completions", json=data)
            if not future.done():
                future.set_result(resp.json())
        except Exception as e:
            if not future.done():
                future.set_exception(e)

        return await future
