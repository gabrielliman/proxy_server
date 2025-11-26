import time
import asyncio
from threading import Lock
from typing import Dict, Any, Optional, List


class ProcessTable:
    def __init__(self):
        self.lock = Lock()
        self.table: Dict[str, Dict[str, Any]] = {}

    def ensure_process(self, program_id: str):
        with self.lock:
            if program_id not in self.table:
                self.table[program_id] = {
                    "service_time_cumulative": 0.0,
                    "service_time_max": 0.0,
                    "waiting_time_cumulative": 0.0,
                    "service_ewma": None,
                    "call_count": 0,
                    "engine_ids": set(),
                    "threads": {},  # call_id -> metadata
                    "most_recent_call_arrival": None,
                    "most_recent_call_completion": None,
                }

    def remove_process(self, program_id: str):
        with self.lock:
            if program_id in self.table:
                del self.table[program_id]

    def record_call_arrival(self, program_id: str, call_id: str, arrival_time: float = None):
        now = arrival_time or time.time()
        self.ensure_process(program_id)
        with self.lock:
            entry = self.table[program_id]
            entry["threads"][call_id] = {
                "arrival_time": now,
                "start_time": None,
                "completion_time": None,
                "enqueue_time": None,
                "dequeue_time": None,
                "waiting_time": 0.0,
                "service_time": 0.0,
                "engine_id": None,
                "state": "waiting",
                #salvar numero de tokens de entrada
            }
            entry["most_recent_call_arrival"] = now

    def record_enqueue(self, program_id: str, call_id: str, enqueue_time: float = None):
        now = enqueue_time or time.time()
        self.ensure_process(program_id)
        with self.lock:
            entry = self.table[program_id]
            th = entry["threads"].get(call_id)
            if th is None:
                th = {
                    "arrival_time": now,
                    "start_time": None,
                    "completion_time": None,
                    "enqueue_time": now,
                    "dequeue_time": None,
                    "waiting_time": 0.0,
                    "service_time": 0.0,
                    "engine_id": None,
                    "state": "waiting",
                }
                entry["threads"][call_id] = th
            else:
                th["enqueue_time"] = now

    def record_dequeue(self, program_id: str, call_id: str, dequeue_time: float = None):
        now = dequeue_time or time.time()
        with self.lock:
            entry = self.table.get(program_id)
            if not entry:
                return
            th = entry["threads"].get(call_id)
            if not th:
                return
            th["dequeue_time"] = now

    def record_call_start(self, program_id: str, call_id: str, engine_id: str = None, start_time: float = None):
        now = start_time or time.time()
        self.ensure_process(program_id)
        with self.lock:
            entry = self.table[program_id]
            th = entry["threads"].get(call_id)
            if th is None:
                # If arrival was not recorded, create a minimal thread record
                th = {
                    "arrival_time": now,
                    "start_time": now,
                    "completion_time": None,
                    "waiting_time": 0.0,
                    "service_time": 0.0,
                    "engine_id": engine_id,
                    "state": "running",
                }
                entry["threads"][call_id] = th
            else:
                th["start_time"] = now
                # compute waiting based on enqueue_time if available, else arrival_time
                base = th.get("enqueue_time") or th.get("arrival_time") or now
                th["waiting_time"] = max(0.0, now - base)
                th["state"] = "running"
                th["engine_id"] = engine_id

            # update process-level aggregates
            entry["waiting_time_cumulative"] += th["waiting_time"]
            if engine_id:
                entry["engine_ids"].add(engine_id)

    def record_call_completion(self, program_id: str, call_id: str, completion_time: float = None):
        now = completion_time or time.time()
        with self.lock:
            entry = self.table.get(program_id)
            if not entry:
                return
            th = entry["threads"].get(call_id)
            if not th:
                return

            th["completion_time"] = now
            if th.get("start_time"):
                service = max(0.0, now - th["start_time"])
            else:
                service = 0.0

            th["service_time"] = service
            th["state"] = "completed"

            entry["service_time_cumulative"] += service
            if service > entry["service_time_max"]:
                entry["service_time_max"] = service

            # update EWMA/count for service time
            entry["call_count"] = entry.get("call_count", 0) + 1
            # leave EWMA update to scheduler with configured alpha if needed; keep simple decay here
            ewma = entry.get("service_ewma")
            if ewma is None:
                entry["service_ewma"] = service
            else:
                # default small-alpha smoothing; scheduler may override using config
                alpha = 0.3
                entry["service_ewma"] = alpha * service + (1 - alpha) * ewma

            entry["most_recent_call_completion"] = now

            # remove engine assignment if no other running thread uses it
            engine = th.get("engine_id")
            if engine and engine in entry["engine_ids"]:
                still_using = any(
                    t.get("engine_id") == engine and t.get("state") == "running"
                    for t in entry["threads"].values()
                )
                if not still_using:
                    entry["engine_ids"].discard(engine)

    def get_process(self, program_id: str):
        with self.lock:
            entry = self.table.get(program_id)
            if not entry:
                return None
            # copy minimal fields while holding the lock
            threads_copy = {cid: dict(th) for cid, th in entry.get("threads", {}).items()}
            engine_ids_copy = list(entry.get("engine_ids", []))
            result = {
                "service_time_cumulative": entry["service_time_cumulative"],
                "service_time_max": entry["service_time_max"],
                "service_ewma": entry.get("service_ewma"),
                "call_count": entry.get("call_count", 0),
                "waiting_time_cumulative": entry["waiting_time_cumulative"],
                "engine_ids": engine_ids_copy,
                "threads": threads_copy,
                "most_recent_call_arrival": entry["most_recent_call_arrival"],
                "most_recent_call_completion": entry["most_recent_call_completion"],
            }

        return result

    def list_processes(self):
        # Take a quick snapshot of keys and shallow copies of entries while holding the lock,
        # then build JSON-serializable copies outside the lock to avoid blocking other threads.
        with self.lock:
            items = list(self.table.items())

        snapshot = {}
        for pid, entry in items:
            # create shallow serializable copies
            threads_copy = {cid: dict(th) for cid, th in entry.get("threads", {}).items()}
            engine_ids_copy = list(entry.get("engine_ids", []))

            snapshot[pid] = {
                "service_time_cumulative": entry.get("service_time_cumulative", 0.0),
                "service_time_max": entry.get("service_time_max", 0.0),
                "waiting_time_cumulative": entry.get("waiting_time_cumulative", 0.0),
                "engine_ids": engine_ids_copy,
                "threads": threads_copy,
                "most_recent_call_arrival": entry.get("most_recent_call_arrival"),
                "most_recent_call_completion": entry.get("most_recent_call_completion"),
            }

        return snapshot

    def prune_stale(self, ttl_seconds: float) -> List[str]:
        """Remove processes that have no waiting/running threads and whose
        most recent call arrival is older than ttl_seconds. Returns list of removed ids.
        """
        now = time.time()
        removed = []
        with self.lock:
            for pid, entry in list(self.table.items()):
                # determine if any active threads exist
                threads = entry.get("threads", {})
                has_active = any(t.get("state") in ("waiting", "running") for t in threads.values())
                last_arrival = entry.get("most_recent_call_arrival")

                if has_active:
                    continue

                if last_arrival is None:
                    # if there are no active threads and no recorded arrival, remove
                    del self.table[pid]
                    removed.append(pid)
                    continue

                if now - last_arrival > ttl_seconds:
                    del self.table[pid]
                    removed.append(pid)

        return removed

    def get_program_stats(self, program_id: str) -> Optional[Dict[str, Any]]:
        """Return a small stats dict useful for schedulers (EWMA, counts, last arrival)."""
        with self.lock:
            entry = self.table.get(program_id)
            if not entry:
                return None
            return {
                "service_ewma": entry.get("service_ewma"),
                "call_count": entry.get("call_count", 0),
                "service_time_cumulative": entry.get("service_time_cumulative", 0.0),
                "most_recent_call_arrival": entry.get("most_recent_call_arrival"),
            }

    def start_async_pruner(self, interval_seconds: float, ttl_seconds: float):
        """Start a background asyncio task that prunes stale processes periodically.
        Safe to call multiple times; only one background task will be running.
        """
        if hasattr(self, "_pruner_task") and self._pruner_task is not None and not self._pruner_task.done():
            return

        async def _loop():
            try:
                while True:
                    await asyncio.sleep(interval_seconds)
                    removed = self.prune_stale(ttl_seconds)
                    if removed:
                        # simple logging to stdout; integrators can hook logging if desired
                        print(f"[PROCESS_TABLE] pruned {len(removed)} processes: {removed}")
            except asyncio.CancelledError:
                return

        self._pruner_task = asyncio.create_task(_loop())

    def stop_async_pruner(self):
        if hasattr(self, "_pruner_task") and self._pruner_task is not None:
            self._pruner_task.cancel()
            self._pruner_task = None


# Module singleton
PROCESS_TABLE = ProcessTable()
