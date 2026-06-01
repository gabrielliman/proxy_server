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
    LOAD_BALANCER_METRICS_LOG_INTERVAL,
    LOAD_BALANCER_THRESHOLD_METRIC,     # e.g., "kv_cache_percent", "waiting", "total", "running"
    LOAD_BALANCER_THRESHOLD_VALUE,      # e.g., 90.0, 10.0, etc.
    LOAD_BALANCER_FALLBACK_STRATEGY,    # e.g., "least-total-load"
)

from routing.process_table import PROCESS_TABLE
    
class LoadBalancerStrategy(ABC):
    """Classe base abstrata. Todas as estratégias DEVEM aceitar 'candidates'."""
    @abstractmethod
    async def select_engine(
        self, 
        program_id: Optional[str], 
        num_input_tokens: int, 
        metrics: Dict[str, Dict], 
        candidates: List[str] = None
    ) -> str:
        pass

    async def on_program_complete(self, program_id: str):
        pass

# --- ESTRATÉGIAS ---
    
class RoundRobinStrategy(LoadBalancerStrategy):
    def __init__(self):
        self._idx = 0
        self._lock = Lock()

    async def select_engine(self, program_id, num_input_tokens, metrics, candidates=None) -> str:
        # Se não houver candidatos específicos, usa todos os backends
        targets = candidates if candidates else ALL_BACKENDS
        if not targets:
            raise RuntimeError("Nenhum backend disponível.")
        with self._lock:
            engine = targets[self._idx % len(targets)]
            self._idx += 1
        return engine

class LeastTotalLoadStrategy(LoadBalancerStrategy):
    async def select_engine(self, program_id, num_input_tokens, metrics, candidates=None) -> str:
        # Filtra métricas apenas para os modelos compatíveis
        target_metrics = {u: m for u, m in metrics.items() if candidates is None or u in candidates}
        if not target_metrics:
            return random.choice(candidates) if candidates else random.choice(ALL_BACKENDS)

        best_engine = min(target_metrics.keys(), 
                         key=lambda e: int(target_metrics[e].get("running", 0)) + int(target_metrics[e].get("waiting", 0)))
        return best_engine

class AutelixStrategy(LoadBalancerStrategy):
    """Estratégia Autellix: Curto -> Carga; Longo -> Afinidade."""
    def __init__(self, short_threshold: int = 2048):
        self.short_threshold = short_threshold

    async def select_engine(self, program_id, num_input_tokens, metrics, candidates=None) -> str:
        target_metrics = {u: m for u, m in metrics.items() if candidates is None or u in candidates}
        
        if not target_metrics:
            return random.choice(candidates) if candidates else random.choice(ALL_BACKENDS)

        # SE REQUISIÇÃO CURTA: Menor carga entre os candidatos
        if num_input_tokens <= self.short_threshold:
            return self._select_least_used(target_metrics)

        # SE REQUISIÇÃO LONGA: Tenta afinidade (Data Locality)
        if program_id:
            proc = PROCESS_TABLE.get_process(program_id)
            pref = proc.get("preferred_engine") if proc else None
            if pref and (candidates is None or pref in candidates):
                return pref

        return self._select_least_used(target_metrics)

    def _select_least_used(self, metrics: Dict[str, Dict]) -> str:
        # Usa local_active_requests da PROCESS_TABLE para maior precisão
        return min(metrics.keys(), key=lambda e: int(metrics[e].get("local_active_requests", 0)))

class KVCacheAwareStrategy(LoadBalancerStrategy):
    """
    Roteia para a engine com menor uso de KV-cache para requisições curtas.
    Requisições longas são fixadas por programa (afinidade), respeitando os candidatos.
    """

    def __init__(self, short_request_threshold: int = 2048):
        self.lock = Lock()
        self.short_request_threshold = short_request_threshold

    async def select_engine(
        self, 
        program_id: Optional[str], 
        num_input_tokens: int, 
        metrics: Dict[str, Dict],
        candidates: List[str] = None  # <--- Adicionado para consistência
    ) -> str:

        # 1. Filtra as métricas para considerar apenas backends que possuem o modelo solicitado
        target_metrics = {
            url: m for url, m in metrics.items() 
            if candidates is None or url in candidates
        }

        if not target_metrics:
            # Fallback: escolhe aleatoriamente entre os candidatos válidos
            return random.choice(candidates) if candidates else random.choice(ALL_BACKENDS)

        # 2. Lógica para Requisições Curtas: Usa o KV-cache como critério
        if num_input_tokens <= self.short_request_threshold:
            return self._select_by_kv_cache(target_metrics)

        # 3. Lógica para Requisições Longas: Tenta afinidade (Data Locality)
        if program_id:
            from routing.process_table import PROCESS_TABLE
            proc = PROCESS_TABLE.get_process(program_id)
            if proc and proc.get("preferred_engine"):
                pref = proc["preferred_engine"]
                # Só utiliza a preferida se ela for compatível com o modelo (estiver nos candidatos)
                if candidates is None or pref in candidates:
                    return pref

        return self._select_by_kv_cache(target_metrics)

    async def on_program_complete(self, program_id: str):
        from routing.process_table import PROCESS_TABLE
        PROCESS_TABLE.clear_preferred_engine(program_id)

    def _select_by_kv_cache(self, metrics: Dict[str, Dict]) -> str:
        # Busca a engine com menor uso entre as métricas já filtradas
        if not metrics:
            return random.choice(ALL_BACKENDS)
            
        best_engine = None
        best_val = float("inf")
        
        for engine, m in metrics.items():
            # Busca o valor de KV cache (vLLM reporta como 'kv_cache_percent')
            kv = m.get("kv_cache_percent")
            
            # Tenta encontrar chaves similares se a principal não existir
            if kv is None:
                for k, v in m.items():
                    if "kv" in k.lower() and "cache" in k.lower():
                        try:
                            kv = float(v)
                            break
                        except: continue
            try:
                val = float(kv) if kv is not None else float("inf")
            except:
                val = float("inf")

            if val < best_val:
                best_val = val
                best_engine = engine
                
        return best_engine or list(metrics.keys())[0]
    
class LeastWaitingStrategy(LoadBalancerStrategy):
    """
    Seleciona a engine com o menor número de requisições na fila (waiting).
    Agora com suporte a filtragem de modelos (candidates).
    """

    async def select_engine(
        self,
        program_id: Optional[str],
        num_input_tokens: int,
        metrics: Dict[str, Dict],
        candidates: List[str] = None  # <--- Adicionado para consistência
    ) -> str:

        # 1. Filtra as métricas para considerar apenas os backends que possuem o modelo
        target_metrics = {
            url: m for url, m in metrics.items() 
            if candidates is None or url in candidates
        }

        if not target_metrics:
            # Fallback seguro: se não houver métricas, escolhe um dos candidatos válidos
            return random.choice(candidates) if candidates else random.choice(ALL_BACKENDS)

        best_engine = None
        best_waiting = float("inf")

        # 2. Busca o menor 'waiting' apenas entre os modelos compatíveis
        for engine, m in target_metrics.items():
            # Tenta obter 'waiting' das métricas do vLLM ou 0 se não disponível
            waiting = int(m.get("waiting", 0))

            if waiting < best_waiting:
                best_waiting = waiting
                best_engine = engine

        return best_engine

class LeastRunningStrategy(LoadBalancerStrategy):
    """
    Select engine with minimum running requests.
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
        best_running = float("inf")

        for engine, m in metrics.items():
            running = int(m.get("running", 0))

            if running < best_running:
                best_running = running
                best_engine = engine

        return best_engine

class LeastKVCacheStrategy(LoadBalancerStrategy):
    """
    Seleciona a engine com o menor uso de KV cache.
    Essencial para evitar fragmentação de memória em contextos longos (20k).
    """

    async def select_engine(
        self,
        program_id: Optional[str],
        num_input_tokens: int,
        metrics: Dict[str, Dict],
        candidates: List[str] = None  # <--- Parâmetro obrigatório para consistência
    ) -> str:

        # 1. Filtramos as métricas para considerar apenas as portas compatíveis com o modelo
        target_metrics = {
            url: m for url, m in metrics.items() 
            if candidates is None or url in candidates
        }

        if not target_metrics:
            # Fallback: escolhe um candidato válido aleatoriamente se não houver métricas
            return random.choice(candidates) if candidates else random.choice(ALL_BACKENDS)

        best_engine = None
        best_kv = float("inf")

        # 2. Busca o menor uso de KV Cache apenas entre os candidatos válidos
        for engine, m in target_metrics.items():
            # vLLM reporta como 'kv_cache_percent' ou 'kv_cache_usage_perc'
            kv = m.get("kv_cache_percent") or m.get("kv_cache_usage_perc")
            
            try:
                kv_val = float(kv) if kv is not None else float("inf")
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


class ThresholdAutellixStrategy(LoadBalancerStrategy):
    """
    Autellix-style with a generalized safety valve threshold.

    Long requests:
      - prefer pinned engine
      - if targeted metric > threshold -> fallback strategy

    Short requests:
      - fallback strategy
    """

    def __init__(
        self,
        short_request_threshold: int,
        threshold_metric: str,
        threshold_value: float,
        fallback_strategy: LoadBalancerStrategy,
    ):
        self.short_request_threshold = short_request_threshold
        self.threshold_metric = threshold_metric
        self.threshold_value = threshold_value
        self.fallback = fallback_strategy

    async def select_engine(
        self,
        program_id: Optional[str],
        num_input_tokens: int,
        metrics: Dict[str, Dict],
        candidates: List[str] = None  # <--- Adicionado para consistência modular
    ) -> str:

        # REQUISIÇÃO CURTA → utiliza a estratégia de recuo (fallback) com os candidatos
        if num_input_tokens <= self.short_request_threshold:
            return await self.fallback.select_engine(
                program_id, num_input_tokens, metrics, candidates=candidates
            )

        # REQUISIÇÃO LONGA → tenta usar a engine fixada (affinity)
        if program_id:
            from routing.process_table import PROCESS_TABLE
            proc = PROCESS_TABLE.get_process(program_id)
            pinned = proc.get("preferred_engine") if proc else None

            # Verifica se a engine fixada é compatível com o modelo atual
            is_valid_candidate = candidates is None or pinned in candidates

            if pinned and pinned in metrics and is_valid_candidate:
                kv = metrics[pinned].get("kv_cache_percent")
                try:
                    kv_val = float(kv)
                except Exception:
                    kv_val = None

                # SE O CACHE ESTIVER OK → mantém na engine fixada
                if kv_val is None or kv_val < self.kv_threshold_percent:
                    return pinned

                # SE O CACHE ESTIVER MUITO ALTO → usa o fallback para achar outra engine válida
                print(f"⚠️ [LB] KV Cache alto ({kv_val}%). Acionando fallback para {program_id}")
                return await self.fallback.select_engine(
                    program_id, num_input_tokens, metrics, candidates=candidates
                )

        # Sem afinidade ou engine não compatível → vai para o fallback
        return await self.fallback.select_engine(
            program_id, num_input_tokens, metrics, candidates=candidates
        )

    async def on_program_complete(self, program_id: str):
        from routing.process_table import PROCESS_TABLE
        PROCESS_TABLE.clear_preferred_engine(program_id)

# --- COORDENADOR (LOAD BALANCER) ---   

class LoadBalancer:
    """Coordinator: metrics collection, logging and strategy dispatch."""

    def __init__(self):
        self.strategy: LoadBalancerStrategy = self._init_strategy()
        self.metrics_cache: Dict[str, Dict] = {engine: {} for engine in ALL_BACKENDS}
        self.metrics_lock = Lock()
        self.last_metricstmux_log = 0.0
        self.metrics_query_task: Optional[asyncio.Task] = None

    def _get_fallback_strategy(self, fallback_name: str) -> LoadBalancerStrategy:
        """Helper to instantiate the chosen fallback strategy."""
        name = fallback_name.lower() if fallback_name else "least-total-load"
        if name == "round-robin":
            return RoundRobinStrategy()
        if name == "least-waiting":
            return LeastWaitingStrategy()
        if name == "least-running":
            return LeastRunningStrategy()
        if name == "least-kv-cache":
            return LeastKVCacheStrategy()
        
        # default fallback
        return LeastTotalLoadStrategy()

    def _init_strategy(self) -> LoadBalancerStrategy:
        name = (LOAD_BALANCER_STRATEGY or "autellix").lower()

        # ------------------------
        # Simple Baselines
        # ------------------------
        if name == "round-robin":
            return RoundRobinStrategy()

        if name == "least-total-load":
            return LeastTotalLoadStrategy()

        if name == "least-waiting":
            return LeastWaitingStrategy()
            
        if name == "least-running":
            return LeastRunningStrategy()

        if name == "least-kv-cache":
            return LeastKVCacheStrategy()

        # ------------------------
        # Original Autellix
        # ------------------------
        if name == "autellix":
            return AutelixStrategy(LOAD_BALANCER_SHORT_REQUEST_THRESHOLD)

        # ------------------------
        # General Threshold Autellix 
        # ------------------------
        if name in ["threshold-autellix", "kv-threshold-autellix"]:
            
            # Fetch from config, or provide safe defaults
            metric = getattr(LOAD_BALANCER_THRESHOLD_METRIC, "lower", lambda: "kv_cache_percent")()
            try:
                value = float(LOAD_BALANCER_THRESHOLD_VALUE)
            except (ValueError, TypeError):
                value = 90.0
                
            fallback_strat = self._get_fallback_strategy(LOAD_BALANCER_FALLBACK_STRATEGY)

            return ThresholdAutellixStrategy(
                short_request_threshold=LOAD_BALANCER_SHORT_REQUEST_THRESHOLD,
                threshold_metric=metric,
                threshold_value=value,
                fallback_strategy=fallback_strat,
            )

        # ------------------------
        # Safe Default
        # ------------------------
        return AutelixStrategy(LOAD_BALANCER_SHORT_REQUEST_THRESHOLD)


      
    async def select_engine(
            self, 
            program_id: Optional[str], 
            num_input_tokens: int,
            candidates: List[str] = None,
            ) -> str:
        metrics = self._get_cached_metrics()
        return await self.strategy.select_engine(
            program_id, 
            num_input_tokens, 
            metrics, 
            candidates=candidates 
        )

    
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