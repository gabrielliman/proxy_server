"""
Modular load balancer with pluggable routing strategies.
Supports Autellix-style (short/long routing) and KV-cache-aware routing.
"""
import random
import asyncio
import time
import httpx
from abc import ABC, abstractmethod
from threading import Lock
from typing import Dict, Optional, List
from config.settings import (
    ALL_BACKENDS,
    LOAD_BALANCER_SHORT_REQUEST_THRESHOLD,
    LOAD_BALANCER_STRATEGY,
    LOAD_BALANCER_ENABLE_METRICS_QUERY,
    REQUEST_TIMEOUT,
)
from routing.process_table import PROCESS_TABLE


class LoadBalancerStrategy(ABC):
    """Abstract base class for load balancing strategies."""

    @abstractmethod
    async def select_engine(
        self, program_id: Optional[str], num_input_tokens: int, metrics: Dict[str, Dict]
    ) -> str:
        """
        Select an engine for the request.

        Args:
            program_id: Program identifier (None for stateless requests)
            num_input_tokens: Number of input tokens in request
            metrics: Dict of {engine_url: {metric_name: value}}

        Returns:
            Selected engine URL
        """
        pass
"""
Modular load balancer with pluggable routing strategies.
Supports Autellix-style (short/long routing) and KV-cache-aware routing.

Provides a background metrics collector that queries engine `/metrics` endpoints
and logs a compact one-line-per-engine summary every `LOAD_BALANCER_METRICS_LOG_INTERVAL` seconds.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import Dict, Optional

import httpx

from config.settings import (
    ALL_BACKENDS,
    LOAD_BALANCER_SHORT_REQUEST_THRESHOLD,
    LOAD_BALANCER_STRATEGY,
    LOAD_BALANCER_ENABLE_METRICS_QUERY,
    REQUEST_TIMEOUT,
    LOAD_BALANCER_METRICS_LOG_INTERVAL,
)

from routing.process_table import PROCESS_TABLE


class LoadBalancerStrategy(ABC):
    """Abstract base class for load balancing strategies."""

    @abstractmethod
    async def select_engine(
        self, program_id: Optional[str], num_input_tokens: int, metrics: Dict[str, Dict]
    ) -> str:
        pass

    async def on_program_complete(self, program_id: str):
        pass

    async def on_call_complete(self, program_id: str, engine_id: str):
        pass


class RoundRobinStrategy(LoadBalancerStrategy):
    """
    Simple round-robin load balancing across ALL_BACKENDS.
    Ignores metrics and distributes requests evenly.
    """

    def __init__(self):
        self._idx = 0
        self._lock = Lock()

    async def select_engine(
        self,
        program_id: Optional[str],
        num_input_tokens: int,
        metrics: Dict[str, Dict],
    ) -> str:
        if not ALL_BACKENDS:
            raise RuntimeError("No backends configured for load balancer")

        with self._lock:
            engine = ALL_BACKENDS[self._idx % len(ALL_BACKENDS)]
            self._idx += 1

        return engine
        
class LeastTotalLoadStrategy(LoadBalancerStrategy):
    """
    Select engine with minimum (running + waiting).
    """

    async def select_engine(
        self,
        program_id: Optional[str],
        num_input_tokens: int,
        metrics: Dict[str, Dict],
    ) -> str:

        if not metrics:
            return random.choice(ALL_BACKENDS)

        best_engine = None
        best_load = float("inf")

        for engine, m in metrics.items():
            running = int(m.get("running", 0))
            waiting = int(m.get("waiting", 0))
            total = running + waiting

            if total < best_load:
                best_load = total
                best_engine = engine

        return best_engine

class LeastWaitingStrategy(LoadBalancerStrategy):
    """
    Select engine with minimum waiting requests.
    """

    async def select_engine(
        self,
        program_id: Optional[str],
        num_input_tokens: int,
        metrics: Dict[str, Dict],
    ) -> str:

        if not metrics:
            return random.choice(ALL_BACKENDS)

        best_engine = None
        best_waiting = float("inf")

        for engine, m in metrics.items():
            waiting = int(m.get("waiting", 0))

            if waiting < best_waiting:
                best_waiting = waiting
                best_engine = engine

        return best_engine

class LeastKVCacheStrategy(LoadBalancerStrategy):
    """
    Select engine with lowest KV cache usage.
    """

    async def select_engine(
        self,
        program_id: Optional[str],
        num_input_tokens: int,
        metrics: Dict[str, Dict],
    ) -> str:

        if not metrics:
            return random.choice(ALL_BACKENDS)

        best_engine = None
        best_kv = float("inf")

        for engine, m in metrics.items():
            kv = m.get("kv_cache_percent")
            try:
                kv_val = float(kv)
            except Exception:
                kv_val = float("inf")

            if kv_val < best_kv:
                best_kv = kv_val
                best_engine = engine

        return best_engine


class AutelixStrategy(LoadBalancerStrategy):
    """
    Autellix-style: short requests -> least-used engine; long requests -> pinned per-program engine.
    """

    def __init__(self, short_request_threshold: int = 2048):
        self.lock = Lock()
        self.short_request_threshold = short_request_threshold

    async def select_engine(
        self, program_id: Optional[str], num_input_tokens: int, metrics: Dict[str, Dict]
    ) -> str:

        # SHORT REQUEST → load balance
        if num_input_tokens <= self.short_request_threshold:
            return self._select_least_used(metrics)

        # LONG REQUEST → data locality
        if program_id:
            proc = PROCESS_TABLE.get_process(program_id)
            if proc and proc.get("preferred_engine"):
                return proc["preferred_engine"]

        engine = self._select_least_used(metrics)
        print("LB SELECT_ENGINE CALLED", program_id, num_input_tokens)

        return engine

    async def on_program_complete(self, program_id: str):
        PROCESS_TABLE.clear_preferred_engine(program_id)


    def _select_least_used(self, metrics: Dict[str, Dict]) -> str:
        if not metrics:
            return random.choice(ALL_BACKENDS)

        min_load = None
        candidates = []

        for engine, m in metrics.items():
            load = int(m.get("local_active_requests", 0))
            if min_load is None or load < min_load:
                min_load = load
                candidates = [engine]
            elif load == min_load:
                candidates.append(engine)

        return random.choice(candidates)



class KVCacheAwareStrategy(LoadBalancerStrategy):
    """
    Route to engine with the lowest KV-cache utilization for short requests.
    Long requests are pinned per-program similar to AutellixStrategy.
    """

    def __init__(self, short_request_threshold: int = 2048):
        self.lock = Lock()
        self.short_request_threshold = short_request_threshold

    async def select_engine(
        self, program_id: Optional[str], num_input_tokens: int, metrics: Dict[str, Dict]
    ) -> str:

        if num_input_tokens <= self.short_request_threshold:
            return self._select_by_kv_cache(metrics)

        if program_id:
            proc = PROCESS_TABLE.get_process(program_id)
            if proc and proc.get("preferred_engine"):
                return proc["preferred_engine"]

        return self._select_by_kv_cache(metrics)

    async def on_program_complete(self, program_id: str):
        PROCESS_TABLE.clear_preferred_engine(program_id)

    def _select_by_kv_cache(self, metrics: Dict[str, Dict]) -> str:
        if not metrics:
            return ALL_BACKENDS[0] if ALL_BACKENDS else "http://localhost:8005"
        best_engine = None
        best_val = float("inf")
        for engine, m in metrics.items():
            kv = m.get("kv_cache_percent")
            if kv is None:
                # try to find any kv-like key
                for k in m.keys():
                    lk = k.lower()
                    if "kv" in lk and "cache" in lk:
                        try:
                            kv = float(m[k])
                        except Exception:
                            kv = None
                        break
            try:
                val = float(kv) if kv is not None else float("inf")
            except Exception:
                val = float("inf")
            if val < best_val:
                best_val = val
                best_engine = engine
        return best_engine or list(metrics.keys())[0]

class KVThresholdAutellixStrategy(LoadBalancerStrategy):
    """
    Autellix-style with KV-cache safety valve.

    Long requests:
      - prefer pinned engine
      - if KV cache > threshold → fallback strategy

    Short requests:
      - fallback strategy
    """

    def __init__(
        self,
        short_request_threshold: int,
        kv_threshold_percent: float,
        fallback_strategy: LoadBalancerStrategy,
    ):
        self.short_request_threshold = short_request_threshold
        self.kv_threshold_percent = kv_threshold_percent
        self.fallback = fallback_strategy

    async def select_engine(
        self,
        program_id: Optional[str],
        num_input_tokens: int,
        metrics: Dict[str, Dict],
    ) -> str:

        # SHORT REQUEST → fallback
        if num_input_tokens <= self.short_request_threshold:
            return await self.fallback.select_engine(
                program_id, num_input_tokens, metrics
            )

        # LONG REQUEST → try pinned engine
        if program_id:
            proc = PROCESS_TABLE.get_process(program_id)
            pinned = proc.get("preferred_engine") if proc else None

            if pinned and pinned in metrics:
                kv = metrics[pinned].get("kv_cache_percent")
                try:
                    kv_val = float(kv)
                except Exception:
                    kv_val = None

                # KV OK → use pinned
                if kv_val is None or kv_val < self.kv_threshold_percent:
                    return pinned

                # KV too high → fallback
                return await self.fallback.select_engine(
                    program_id, num_input_tokens, metrics
                )

        # no pin → fallback
        return await self.fallback.select_engine(
            program_id, num_input_tokens, metrics
        )

    async def on_program_complete(self, program_id: str):
        PROCESS_TABLE.clear_preferred_engine(program_id)


class LoadBalancer:
    """Coordinator: metrics collection, logging and strategy dispatch."""

    def __init__(self):
        self.strategy: LoadBalancerStrategy = self._init_strategy()
        self.metrics_cache: Dict[str, Dict] = {engine: {} for engine in ALL_BACKENDS}
        self.metrics_lock = Lock()
        self.last_metrics_log = 0.0
        self.metrics_query_task: Optional[asyncio.Task] = None

    def _init_strategy(self) -> LoadBalancerStrategy:
        name = (LOAD_BALANCER_STRATEGY or "autellix").lower()

        # ------------------------
        # Baselines simples
        # ------------------------
        if name == "round-robin":
            return RoundRobinStrategy()

        if name == "least-total-load":
            return LeastTotalLoadStrategy()

        if name == "least-waiting":
            return LeastWaitingStrategy()

        if name == "least-kv-cache":
            return LeastKVCacheStrategy()

        # ------------------------
        # Autellix original
        # ------------------------
        if name == "autellix":
            return AutelixStrategy(LOAD_BALANCER_SHORT_REQUEST_THRESHOLD)

        # ------------------------
        # KV-aware simples (já existente)
        # ------------------------
        if name == "kv-cache":
            return KVCacheAwareStrategy(LOAD_BALANCER_SHORT_REQUEST_THRESHOLD)

        # ------------------------
        # Autellix + KV threshold (fallback modular)
        # ------------------------
        if name == "kv-threshold-autellix":
            # fallback pode ser trocado facilmente
            fallback = LeastTotalLoadStrategy() #por enquanto esta hardcoded, mas da para trocar pelas outras estrategias de escalonamento definidas

            return KVThresholdAutellixStrategy(
                short_request_threshold=LOAD_BALANCER_SHORT_REQUEST_THRESHOLD,
                kv_threshold_percent=90.0,
                fallback_strategy=fallback,
            )

        # ------------------------
        # Default seguro
        # ------------------------
        return AutelixStrategy(LOAD_BALANCER_SHORT_REQUEST_THRESHOLD)


    async def select_engine(self, program_id: Optional[str], num_input_tokens: int) -> str:
        metrics = self._get_cached_metrics()
        return await self.strategy.select_engine(program_id, num_input_tokens, metrics)

    async def on_program_complete(self, program_id: str):
        await self.strategy.on_program_complete(program_id)

    async def on_call_complete(self, program_id: str, engine_id: str):
        await self.strategy.on_call_complete(program_id, engine_id)

    def _get_cached_metrics(self) -> Dict[str, Dict]:
        with self.metrics_lock:
            return {engine: dict(m) for engine, m in self.metrics_cache.items()}

    async def start_metrics_collector(self):
        if not LOAD_BALANCER_ENABLE_METRICS_QUERY:
            return

        async def loop():
            try:
                while True:
                    await self._query_engine_metrics()
                    await self._log_metrics_if_needed()
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                return

        self.metrics_query_task = asyncio.create_task(loop())

    async def stop_metrics_collector(self):
        if self.metrics_query_task:
            self.metrics_query_task.cancel()
            try:
                await self.metrics_query_task
            except asyncio.CancelledError:
                pass
            self.metrics_query_task = None

    async def _query_engine_metrics(self):
        tasks = [self._fetch_engine_metrics(e) for e in ALL_BACKENDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        with self.metrics_lock:
            for engine, res in zip(ALL_BACKENDS, results):
                if isinstance(res, dict):
                    self.metrics_cache.setdefault(engine, {})
                    self.metrics_cache[engine].update(res)
                else:
                    self.metrics_cache.setdefault(engine, {})["query_error"] = str(res)

            # include local workload counts from PROCESS_TABLE if available
            if hasattr(PROCESS_TABLE, "get_all_engine_workloads"):
                workloads = PROCESS_TABLE.get_all_engine_workloads()
            else:
                workloads = {}
            for engine in ALL_BACKENDS:
                self.metrics_cache.setdefault(engine, {})
                self.metrics_cache[engine]["local_active_requests"] = workloads.get(engine, 0)

    async def _fetch_engine_metrics(self, engine_url: str) -> Dict:
        """
        Query /metrics from vLLM and return a clean dictionary with correct values.
        This strictly parses only canonical vLLM counters so the results exactly
        match the internal vLLM logging behavior.
        """

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.get(f"{engine_url}/metrics", timeout=2.0)

                if resp.status_code != 200:
                    return {"error": f"Status {resp.status_code}"}

                text = resp.text

        except Exception as e:
            return {"error": str(e)}

        # --------------------------
        # Parse Prometheus text
        # --------------------------
        parsed = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # remove labels
            if "#" in line:
                line = line.split("#", 1)[0].strip()

            parts = line.split()
            if len(parts) < 2:
                continue

            metric_name = parts[0]
            value = parts[-1]

            try:
                parsed[metric_name] = float(value)
            except Exception:
                parsed[metric_name] = value

        # --------------------------
        # Strict metric extraction
        # --------------------------
        result = {}

        prefix_hits_total = None
        prefix_queries_total = None

        prompt_tokens_total = None
        generation_tokens_total = None
        process_start_time = None

        for raw_key, v in parsed.items():
            base = raw_key.split("{", 1)[0]  # strip labels

            # ACTIVE REQUESTS
            if base == "vllm:num_requests_running":
                result["running"] = int(v)

            elif base == "vllm:num_requests_waiting":
                result["waiting"] = int(v)

            # KV CACHE USAGE (0-1) → convert to %
            elif base == "vllm:kv_cache_usage_perc":
                result["kv_cache_percent"] = float(v) * 100.0

            # PREFIX CACHE COUNTERS
            elif base == "vllm:prefix_cache_hits_total":
                prefix_hits_total = float(v)

            elif base == "vllm:prefix_cache_queries_total":
                prefix_queries_total = float(v)

            # TOKEN TOTAL COUNTERS
            elif base == "vllm:prompt_tokens_total":
                prompt_tokens_total = float(v)

            elif base == "vllm:generation_tokens_total":
                generation_tokens_total = float(v)

            # PROCESS START TIME
            elif base == "process_start_time_seconds":
                process_start_time = float(v)

            # ignore histogram buckets, created, sum, count
            elif any(s in base for s in ["_sum", "_count", "_bucket", "_created"]):
                continue

            else:
                # keep other metrics with clean names
                result[base] = v

        # --------------------------
        # Compute prefix hit rate
        # --------------------------
        if prefix_hits_total is not None and prefix_queries_total:
            result["prefix_cache_hits"] = prefix_hits_total
            result["prefix_cache_queries"] = prefix_queries_total
            result["prefix_cache_hit_rate"] = prefix_hits_total / prefix_queries_total

        # --------------------------
        # Compute throughput (tokens/sec)
        # --------------------------
        if process_start_time:
            uptime = max(1.0, time.time() - process_start_time)

            if prompt_tokens_total is not None:
                result["prompt_tps"] = prompt_tokens_total / uptime

            if generation_tokens_total is not None:
                result["gen_tps"] = generation_tokens_total / uptime

        return result

    async def _log_metrics_if_needed(self):
        now_ts = time.time()
        if now_ts - self.last_metrics_log < (LOAD_BALANCER_METRICS_LOG_INTERVAL or 5):
            return
        with self.metrics_lock:
            snapshot = {engine: dict(m) for engine, m in self.metrics_cache.items()}
        self.last_metrics_log = now_ts
        now = time.strftime("%m-%d %H:%M:%S")

        for idx, engine in enumerate(ALL_BACKENDS):
            em = snapshot.get(engine, {}) or {}

            # flexible find helpers
            def find_metric(keys_subs):
                for k, v in em.items():
                    lk = k.lower()
                    if all(sub in lk for sub in keys_subs):
                        return v
                return None

            running = em.get("running") or em.get("local_active_requests") or em.get("active") or 0
            waiting = em.get("queue_size") or em.get("waiting") or 0

            kv = em.get("kv_cache_percent") or find_metric(["kv", "cache"]) or None
            prefix_hit = em.get("prefix_cache_hit_rate") or find_metric(["prefix", "hit"]) or None

            def as_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            kv_f = as_float(kv)
            prefix_f = as_float(prefix_hit)

            # normalize prefix hit: support either fraction (0..1) or percentage (0..100)
            prefix_pct = None
            if prefix_f is None:
                prefix_pct = None
            else:
                try:
                    if 0.0 <= prefix_f <= 1.0:
                        prefix_pct = prefix_f * 100.0
                    elif 1.0 < prefix_f <= 100.0:
                        prefix_pct = prefix_f
                    else:
                        # unexpected large value: cap at 100
                        prefix_pct = min(float(prefix_f), 100.0)
                except Exception:
                    prefix_pct = None

            running_s = f"{int(running)} reqs" if isinstance(running, (int, float)) else str(running)
            waiting_s = f"{int(waiting)} reqs" if isinstance(waiting, (int, float)) else str(waiting)
            kv_s = f"{kv_f:.1f}%" if kv_f is not None else "N/A"
            prefix_s = f"{prefix_pct:.1f}%" if prefix_pct is not None else "N/A"

            # Print compact log without throughput (requested)
            print(
                f"(LoadBalancer) INFO {now} [load_balancer] Engine {idx:03d}: "
                f"Running: {running_s}, Waiting: {waiting_s}, GPU KV cache usage: {kv_s}, "
                f"Prefix cache hit rate: {prefix_s}"
            )


# module-level instance used by dispatchers and main
LOAD_BALANCER = LoadBalancer()
