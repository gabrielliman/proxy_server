import asyncio
import time

from routing import scheduler_plas, queue_manager
from routing.process_table import PROCESS_TABLE
from config import settings


def test_map_priority_to_queue_index_basic():
    buckets = settings.DISCRETIZED_PRIORITY_BUCKETS
    base = settings.DISCRETIZED_PRIORITY_BASE

    # lowest priority should map to highest index
    idx_high = scheduler_plas.map_priority_to_queue_index(base * 0.9, buckets, base)
    assert 0 <= idx_high < buckets

    # zero priority maps to bucket 0
    idx_zero = scheduler_plas.map_priority_to_queue_index(0.0, buckets, base)
    assert idx_zero == 0


def test_discretized_queue_dequeue_order():
    async def _run():
        dq = queue_manager.DiscretizedQueueWrapper(3)
        await dq.put(2, {"id": "low"})
        await dq.put(1, {"id": "mid"})
        await dq.put(0, {"id": "high"})

        a = await dq.get()
        b = await dq.get()
        c = await dq.get()

        assert a["id"] == "high"
        assert b["id"] == "mid"
        assert c["id"] == "low"

    asyncio.run(_run())


def test_anti_starvation_promotion_and_reset():
    pid = "prom-test-prog"
    PROCESS_TABLE.remove_process(pid)
    PROCESS_TABLE.ensure_process(pid)

    with PROCESS_TABLE.lock:
        PROCESS_TABLE.table[pid]["waiting_time_cumulative"] = 10.0
        PROCESS_TABLE.table[pid]["service_time_cumulative"] = 1.0

    assert PROCESS_TABLE.should_promote_to_q1(pid, beta_threshold=0.5)

    PROCESS_TABLE.reset_wait_and_service_for_promotion(pid)
    with PROCESS_TABLE.lock:
        assert PROCESS_TABLE.table[pid]["waiting_time_cumulative"] == 0.0
        assert PROCESS_TABLE.table[pid]["service_time_cumulative"] == 0.0
