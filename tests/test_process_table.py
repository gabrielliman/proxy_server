import time
from routing.process_table import PROCESS_TABLE


def test_basic_lifecycle():
    pid = "prog-test"
    cid = "call-1"

    # ensure clean state
    PROCESS_TABLE.remove_process(pid)

    PROCESS_TABLE.record_call_arrival(pid, cid)
    p = PROCESS_TABLE.get_process(pid)
    assert p is not None
    assert p["most_recent_call_arrival"] is not None
    assert cid in p["threads"]

    # start
    start = time.time()
    PROCESS_TABLE.record_call_start(pid, cid, engine_id="engine-A", start_time=start)
    p = PROCESS_TABLE.get_process(pid)
    t = p["threads"][cid]
    assert t["state"] == "running"
    assert "engine-A" in p["engine_ids"]

    # complete
    end = start + 0.05
    PROCESS_TABLE.record_call_completion(pid, cid, completion_time=end)
    p = PROCESS_TABLE.get_process(pid)
    t = p["threads"][cid]
    assert t["state"] == "completed"
    # allow small floating-point rounding; ensure service time recorded
    assert p["service_time_cumulative"] >= 0.048

    # cleanup
    PROCESS_TABLE.remove_process(pid)
    assert PROCESS_TABLE.get_process(pid) is None
