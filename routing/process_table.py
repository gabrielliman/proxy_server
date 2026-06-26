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
                    "call_count": 0,

                    "active_thread_count": 0,  # <-- ADD: Threads rodando ou na fila agora
                    "max_parallelism": 0,      # <-- ADD: Pico máximo de threads simultâneas

                    # KV token-time metric (d*c = pd + d²/2)
                    "kv_token_time_cumulative": 0.0,

                    "longest_critical_path": 0.0,
                    "longest_kv_critical_path": 0.0,

                    # NEW (Autellix Alg.2, line 6)
                    "preferred_engines": [],

                    "engine_ids": set(),
                    "engine_request_counts": {},
                    "threads": {},
                    "most_recent_call_arrival": None,
                    "most_recent_call_completion": None,
                }


    def remove_process(self, program_id: str):
        with self.lock:
            if program_id in self.table:
                del self.table[program_id]

    def record_call_arrival(self, program_id: str, call_id: str, arrival_time: float = None, prefill_tokens: int = None):
        now = arrival_time or time.time()
        self.ensure_process(program_id)
        with self.lock:
            entry = self.table[program_id]

            if call_id not in entry["threads"]:
                entry["active_thread_count"] = entry.get("active_thread_count", 0) + 1
                entry["max_parallelism"] = max(entry.get("max_parallelism", 0), entry["active_thread_count"])

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
                "inherited_critical_path": entry.get("longest_critical_path", 0.0),
                "inherited_kv_critical_path": entry.get("longest_kv_critical_path", 0.0),
                # KV token tracking
                "prefill_tokens": prefill_tokens,
                "decode_tokens": None,
                "kv_token_time": None,
            }
            entry["most_recent_call_arrival"] = now

    def record_enqueue(self, program_id: str, call_id: str, enqueue_time: float = None):
        now = enqueue_time or time.time()
        self.ensure_process(program_id)
        with self.lock:
            entry = self.table[program_id]
            th = entry["threads"].get(call_id)

            entry["active_thread_count"] = entry.get("active_thread_count", 0) + 1
            entry["max_parallelism"] = max(entry.get("max_parallelism", 0), entry["active_thread_count"])
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

    def record_call_start(
        self,
        program_id: str,
        call_id: str,
        engine_id: str = None,
        start_time: float = None,
    ):
        now = start_time or time.time()
        self.ensure_process(program_id)

        with self.lock:
            entry = self.table[program_id]

            th = entry["threads"].get(call_id)
            if th is None:
                entry["active_thread_count"] = entry.get("active_thread_count", 0) + 1
                entry["max_parallelism"] = max(entry.get("max_parallelism", 0), entry["active_thread_count"])
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
                base = th.get("enqueue_time") or th.get("arrival_time") or now
                th["waiting_time"] = max(0.0, now - base)
                th["state"] = "running"
                th["engine_id"] = engine_id

            entry["waiting_time_cumulative"] += th["waiting_time"]

            if engine_id:
                entry["engine_ids"].add(engine_id)
                
                # <-- ADD THIS: Increment the request count for this specific engine
                entry["engine_request_counts"][engine_id] = entry.get("engine_request_counts", {}).get(engine_id, 0) + 1

                # 🔥 Autellix pinning rule:
                # first long call determines program engine
                if not entry.get("preferred_engines"):
                    entry["preferred_engines"] = [engine_id]

    def add_preferred_engine(self, program_id: str, engine_id: str):
        """Adiciona uma nova engine à lista de favoritas (Spillover)."""
        with self.lock:
            entry = self.table.get(program_id)
            if entry:
                if engine_id not in entry["preferred_engines"]:
                    entry["preferred_engines"].append(engine_id)

    def clear_preferred_engines(self, program_id: str):
        with self.lock:
            entry = self.table.get(program_id)
            if entry:
                entry["preferred_engines"] = []

    def record_kv_tokens(self, program_id: str, call_id: str, prefill_tokens: int, decode_tokens: int):
        """Record KV token counts and calculate KV token-time.
        
        KV token-time formula: d*c = pd + d²/2
        Where:
            p = prefill_tokens (input tokens)
            d = decode_tokens (output tokens)
        """
        with self.lock:
            entry = self.table.get(program_id)
            if not entry:
                return
            th = entry["threads"].get(call_id)
            if not th:
                return
            
            # Store token counts
            th["prefill_tokens"] = prefill_tokens
            th["decode_tokens"] = decode_tokens
            
            # Calculate KV token-time: pd + d²/2
            kv_time = (prefill_tokens * decode_tokens) + (decode_tokens ** 2) / 2
            th["kv_token_time"] = kv_time
            
            # Update cumulative
            entry["kv_token_time_cumulative"] = entry.get("kv_token_time_cumulative", 0.0) + kv_time

    def record_call_completion(self, program_id: str, call_id: str, completion_time: float = None, output_tokens: int = None):
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

            if th.get("state") != "completed":
                entry["active_thread_count"] = max(0, entry.get("active_thread_count", 0) - 1)

            th["service_time"] = service
            th["state"] = "completed"

            entry["service_time_cumulative"] += service
            if service > entry["service_time_max"]:
                entry["service_time_max"] = service

            thread_path = th.get("inherited_critical_path", 0.0) + service
            entry["longest_critical_path"] = max(entry.get("longest_critical_path", 0.0), thread_path)

            entry["call_count"] = entry.get("call_count", 0) + 1

            prefill_tokens = th.get("prefill_tokens")
            decode_tokens = output_tokens if output_tokens is not None else th.get("decode_tokens")
            
            if prefill_tokens is not None and decode_tokens is not None:
                # Calculate KV token-time: pd + d²/2
                kv_time = (prefill_tokens * decode_tokens) + (decode_tokens ** 2) / 2
                th["kv_token_time"] = kv_time
                th["decode_tokens"] = decode_tokens
                
                # Update cumulative
                entry["kv_token_time_cumulative"] = entry.get("kv_token_time_cumulative", 0.0) + kv_time
                
                # ATLAS: Update global critical path scalar for KV token time
                kv_thread_path = th.get("inherited_kv_critical_path", 0.0) + kv_time
                entry["longest_kv_critical_path"] = max(entry.get("longest_kv_critical_path", 0.0), kv_thread_path)

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
                "call_count": entry.get("call_count", 0),
                "max_parallelism": entry.get("max_parallelism", 0),
                "waiting_time_cumulative": entry["waiting_time_cumulative"],
                "preferred_engines": entry.get("preferred_engines", []),
                "engine_ids": engine_ids_copy,
                "engine_request_counts": dict(entry.get("engine_request_counts", {})),
                "threads": threads_copy,
                "most_recent_call_arrival": entry["most_recent_call_arrival"],
                "most_recent_call_completion": entry["most_recent_call_completion"],
                # KV token-time metrics
                "kv_token_time_cumulative": entry.get("kv_token_time_cumulative", 0.0),
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
                "call_count": entry.get("call_count", 0.0),
                "max_parallelism": entry.get("max_parallelism", 0),
                "kv_token_time_cumulative": entry.get("kv_token_time_cumulative", 0.0),
                "longest_critical_path": entry.get("longest_critical_path", 0.0),
                "longest_kv_critical_path": entry.get("longest_kv_critical_path", 0.0),
                "preferred_engines": entry.get("preferred_engines", []),
                "engine_ids": engine_ids_copy,
                "engine_request_counts": dict(entry.get("engine_request_counts", {})),
                "threads": threads_copy,
                "most_recent_call_arrival": entry.get("most_recent_call_arrival"),
                "most_recent_call_completion": entry.get("most_recent_call_completion"),
            }

        return snapshot
    
    def sum_processes(self):
        # Take a quick snapshot of keys and shallow copies of entries while holding the lock,
        # then build JSON-serializable copies outside the lock to avoid blocking other threads.
        with self.lock:
            items = list(self.table.items())

        snapshot = {}
        for pid, entry in items:
            # create shallow serializable copies
            engine_ids_copy = list(entry.get("engine_ids", []))

            snapshot[pid] = {
                "service_time_cumulative": entry.get("service_time_cumulative", 0.0),
                "service_time_max": entry.get("service_time_max", 0.0),
                "waiting_time_cumulative": entry.get("waiting_time_cumulative", 0.0),
                "call_count": entry.get("call_count", 0.0),
                "max_parallelism": entry.get("max_parallelism", 0),
                "kv_token_time_cumulative": entry.get("kv_token_time_cumulative", 0.0),
                "longest_critical_path": entry.get("longest_critical_path", 0.0),
                "longest_kv_critical_path": entry.get("longest_kv_critical_path", 0.0),
                "preferred_engines": entry.get("preferred_engines", []),
                "engine_ids": engine_ids_copy,
                "engine_request_counts": dict(entry.get("engine_request_counts", {})),
                "most_recent_call_arrival": entry.get("most_recent_call_arrival"),
                "most_recent_call_completion": entry.get("most_recent_call_completion"),
            }

        return snapshot
    
    def get_waiting_time(self, program_id: str) -> Optional[float]:
        """Return total waiting time for a program, or None if not found."""
        with self.lock:
            entry = self.table.get(program_id)
            if not entry:
                return None
            return entry.get("waiting_time_cumulative", 0.0)
    
    def get_service_time(self, program_id: str) -> Optional[float]:
        """Return total service time for a program, or None if not found."""
        with self.lock:
            entry = self.table.get(program_id)
            if not entry:
                return None
            return entry.get("service_time_cumulative", 0.0)

    def get_incomplete_requests_count(self) -> int:
        """Retorna o número total de requisições (threads) que ainda não foram concluídas."""
        count = 0
        with self.lock:
            for entry in self.table.values():
                for th in entry.get("threads", {}).values():
                    if th.get("state") in ("waiting", "running"):
                        count += 1
        return count
    
    def get_waiting_requests_count(self) -> int:
        """Retorna o número total de requisições (threads) que estão esperando."""
        count = 0
        with self.lock:
            for entry in self.table.values():
                for th in entry.get("threads", {}).values():
                    if th.get("state") == "waiting":
                        count += 1
        return count
    
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

    def get_thread_stats(self, program_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the inherited critical path metrics for a specific thread (ATLAS)."""
        with self.lock:
            entry = self.table.get(program_id)
            if not entry:
                return None
            th = entry.get("threads", {}).get(call_id)
            if not th:
                return None
            return {
                "inherited_critical_path": th.get("inherited_critical_path", 0.0),
                "inherited_kv_critical_path": th.get("inherited_kv_critical_path", 0.0),
            }

    def get_program_stats(self, program_id: str) -> Optional[Dict[str, Any]]:
        """Return a small stats dict useful for schedulers (counts, last arrival)."""
        with self.lock:
            entry = self.table.get(program_id)
            if not entry:
                return None
            return {
                "call_count": entry.get("call_count", 0),
                "service_time_cumulative": entry.get("service_time_cumulative", 0.0),
                "most_recent_call_arrival": entry.get("most_recent_call_arrival"),
                # KV token-time metrics
                "kv_token_time_cumulative": entry.get("kv_token_time_cumulative", 0.0),
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

    def should_promote_to_q1(self, program_id: str, beta_threshold: float = 0.5) -> bool:
        """Check if program should be promoted to highest priority queue (Q1).
        Promotion occurs when W_total / T_total >= beta_threshold.
        - W_total = waiting_time_cumulative (time spent waiting for backend)
        - T_total = service_time_cumulative (time spent executing on backend)
        """
        with self.lock:
            entry = self.table.get(program_id)
            if not entry:
                return False
            
            w_total = entry.get("waiting_time_cumulative", 0.0)
            t_total = entry.get("service_time_cumulative", 0.0)
            
            if t_total == 0:
                # No service time yet; if waiting, consider promoting
                return w_total > 0
            
            ratio = w_total / t_total
            return ratio >= beta_threshold

    def reset_wait_and_service_for_promotion(self, program_id: str):
        """Reset per-call metrics (W_c, T_c) when promoting to Q1.
        Per Autellix paper, promotion resets the call-level waiting and service times,
        but keeps the program-level aggregates (W_p, T_p) unchanged.
        For simplicity, we reset the cumulative counters to encourage fairness.
        """
        with self.lock:
            entry = self.table.get(program_id)
            if not entry:
                return
            
            # Reset cumulative waiting and service time to restart the ratio check
            entry["waiting_time_cumulative"] = 0.0
            entry["service_time_cumulative"] = 0.0


# Module singleton
PROCESS_TABLE = ProcessTable()
