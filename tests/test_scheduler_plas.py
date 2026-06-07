import time
import asyncio
import math

from routing import scheduler_plas
from routing import queue_manager
from routing.process_table import PROCESS_TABLE
from config import settings


def test_compute_priority_prefers_smaller_ewma():
    p_small = "prog-small"
    p_big = "prog-big"

    PROCESS_TABLE.remove_process(p_small)
    PROCESS_TABLE.remove_process(p_big)
    PROCESS_TABLE.ensure_process(p_small)
    PROCESS_TABLE.ensure_process(p_big)

    now = time.time()
    # set EWMA and recent arrival times under lock
    with PROCESS_TABLE.lock:
        PROCESS_TABLE.table[p_small]["service_ewma"] = 0.1
        PROCESS_TABLE.table[p_small]["most_recent_call_arrival"] = now
        PROCESS_TABLE.table[p_big]["service_ewma"] = 1.0
        PROCESS_TABLE.table[p_big]["most_recent_call_arrival"] = now

    pr_small, _ = scheduler_plas.compute_priority(p_small, "c1")
    pr_big, _ = scheduler_plas.compute_priority(p_big, "c2")

    assert pr_small <= pr_big


def test_aging_increases_priority():
    p_old = "prog-old"
    p_new = "prog-new"
    PROCESS_TABLE.remove_process(p_old)
    PROCESS_TABLE.remove_process(p_new)
    PROCESS_TABLE.ensure_process(p_old)
    PROCESS_TABLE.ensure_process(p_new)

    now = time.time()
    with PROCESS_TABLE.lock:
        PROCESS_TABLE.table[p_old]["service_ewma"] = 1.0
        PROCESS_TABLE.table[p_old]["most_recent_call_arrival"] = now - 300.0
        PROCESS_TABLE.table[p_new]["service_ewma"] = 1.0
        PROCESS_TABLE.table[p_new]["most_recent_call_arrival"] = now

    pr_old, _ = scheduler_plas.compute_priority(p_old, "c1")
    pr_new, _ = scheduler_plas.compute_priority(p_new, "c2")

    assert pr_old <= pr_new


def test_priority_queue_tiebreaker():
    async def _run():
        pq = queue_manager.PriorityQueueWrapper()
        await pq.put(1.0, {"id": 1})
        await pq.put(1.0, {"id": 2})

        a = await pq.get()
        b = await pq.get()
        assert a["id"] == 1
        assert b["id"] == 2

    asyncio.run(_run())


def test_update_on_completion_updates_ewma():
    pid = "ewma-prog"
    PROCESS_TABLE.remove_process(pid)
    PROCESS_TABLE.ensure_process(pid)

    # ensure no ewma
    stats = PROCESS_TABLE.get_program_stats(pid)
    assert stats["service_ewma"] is None

    scheduler_plas.update_on_completion(pid, 0.5)
    stats = PROCESS_TABLE.get_program_stats(pid)
    assert math.isclose(stats["service_ewma"], 0.5, rel_tol=1e-6)

    # second update should apply EWMA with alpha from settings
    alpha = settings.PLAS_EWMA_ALPHA
    scheduler_plas.update_on_completion(pid, 0.2)
    stats = PROCESS_TABLE.get_program_stats(pid)
    expected = alpha * 0.2 + (1 - alpha) * 0.5
    assert math.isclose(stats["service_ewma"], expected, rel_tol=1e-6)
