import asyncio
from typing import Dict

from config.settings import ALL_BACKENDS


backend_queues: Dict[str, asyncio.Queue] = {}


def init_queues():
    """
    Inicializa filas para todos os backends.
    Chamada automática na primeira importação do dispatcher.
    """
    global backend_queues

    if backend_queues:
        return

    backend_queues = {
        backend: asyncio.Queue()
        for backend in ALL_BACKENDS
    }


def get_queue(backend: str) -> asyncio.Queue:
    """Retorna a fila do backend (já inicializada)."""
    return backend_queues[backend]
