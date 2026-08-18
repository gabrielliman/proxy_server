import asyncio
import random
import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import Dict, Optional, List
from collections import deque
import httpx

from config.settings import (
    ALL_BACKENDS,
    BACKEND_PARALLELISM,
    LOAD_BALANCER_SHORT_REQUEST_THRESHOLD,
    LOAD_BALANCER_STRATEGY,
    LOAD_BALANCER_ENABLE_METRICS_QUERY,
    REQUEST_TIMEOUT,
    LOAD_BALANCER_METRICS_LOG_INTERVAL,
    LOAD_BALANCER_THRESHOLD_METRIC,     # e.g., "kv_cache_percent", "waiting", "total", "running"
    LOAD_BALANCER_THRESHOLD_VALUE,      # e.g., 90.0, 10.0, etc.
    LOAD_BALANCER_FALLBACK_STRATEGY,    # e.g., "least-total-load"
)
import config.settings as settings

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
    async def select_engine(self, program_id: Optional[str], num_input_tokens: int, metrics: Dict[str, Dict]) -> str:
        if not metrics:
            return random.choice(ALL_BACKENDS)

        best_engine = None
        best_load = float("inf")

        for engine, m in metrics.items():
            total = int(m.get("pt_running", 0))  # <-- Mudança aqui

            if total < best_load:
                best_load = total
                best_engine = engine

        return best_engine or random.choice(ALL_BACKENDS)

class LeastWaitingStrategy(LoadBalancerStrategy):
    #pega metricas de jeito ruim e lento
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

class LeastRunningStrategy(LoadBalancerStrategy):
    #pega metricas de jeito ruim e lento
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
    #pega metricas de jeito ruim e lento, unico jeito possivel
    
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
            preferred = proc.get("preferred_engines") if proc else None
            if preferred and len(preferred) > 0:
                return preferred[0]

        engine = self._select_least_used(metrics)

        return engine

    async def on_program_complete(self, program_id: str):
        PROCESS_TABLE.clear_preferred_engine(program_id)


    def _select_least_used(self, metrics: Dict[str, Dict]) -> str:
            if not metrics:
                return random.choice(ALL_BACKENDS)

            min_load = None
            candidates = []

            for engine, m in metrics.items():
                load = int(m.get("pt_running", 0)) # <-- Mudança aqui
                if min_load is None or load < min_load:
                    min_load = load
                    candidates = [engine]
                elif load == min_load:
                    candidates.append(engine)

            return random.choice(candidates) if candidates else random.choice(ALL_BACKENDS)

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
    ) -> str:

        # SHORT REQUEST → fallback
        if num_input_tokens <= self.short_request_threshold:
            return await self.fallback.select_engine(
                program_id, num_input_tokens, metrics
            )

        # LONG REQUEST → try pinned engine
        if program_id:
            proc = PROCESS_TABLE.get_process(program_id)
            preferred = proc.get("preferred_engines") if proc else None
            pinned = preferred[0] if preferred and len(preferred) > 0 else None

            if pinned and pinned in metrics:
                m = metrics[pinned]
                
                # Special aggregation case for "total"
                if self.threshold_metric == "total":
                    metric_val = float(m.get("pt_running", 0))
                else:
                    val = m.get(self.threshold_metric)
                    try:
                        metric_val = float(val) if val is not None else None
                    except Exception:
                        metric_val = None

                # Metric OK → use pinned
                if metric_val is None or metric_val < self.threshold_value:
                    return pinned

                # Metric too high → fallback
                return await self.fallback.select_engine(
                    program_id, num_input_tokens, metrics
                )

        # no pin → fallback
        return await self.fallback.select_engine(
            program_id, num_input_tokens, metrics
        )

    async def on_program_complete(self, program_id: str):
        PROCESS_TABLE.clear_preferred_engine(program_id)

class BaseDynamicAutellixStrategy(LoadBalancerStrategy):
    """Classe base contendo a lógica compartilhada para Autellix Dinâmico."""
    
    def __init__(
        self, 
        short_request_threshold: int = 2048,
        threshold_metric: str = "total",
    ):
        self.short_request_threshold = short_request_threshold
        self.threshold_metric = threshold_metric
        self.lock = Lock()

    def _is_engine_full(self, engine: str, metrics: Dict[str, Dict]) -> bool:
        capacity = BACKEND_PARALLELISM.get(engine, 10)
        m = metrics.get(engine, {})
        if self.threshold_metric == "total":
            metric_val = float(m.get("pt_running", 0))
        else:
            val = m.get(self.threshold_metric)
            try:
                metric_val = float(val) if val is not None else None
            except Exception:
                metric_val = None

        # If we can't parse the metric, default to not full.
        # Otherwise, check if the metric meets or exceeds the threshold.
        if metric_val is None:
            return False
            
        return metric_val >= capacity

    def _select_least_used_global(self, metrics: Dict[str, Dict]) -> str:
        """Seleciona a engine mais ociosa do cluster inteiro."""
        if not metrics:
            return random.choice(ALL_BACKENDS)

        min_load = float("inf")
        candidates = []

        for engine, m in metrics.items():
            if self.threshold_metric == "total":
                load = float(m.get("pt_running", 0)) # <-- Mudança aqui
            else:
                val = m.get(self.threshold_metric)
                try:
                    load = float(val) if val is not None else float("inf")
                except Exception:
                    load = float("inf")

            if load < min_load:
                min_load = load
                candidates = [engine]
            elif load == min_load:
                candidates.append(engine)

        return random.choice(candidates) if candidates else random.choice(ALL_BACKENDS)
    
    def _select_least_used_subset(self, subset: List[str], metrics: Dict[str, Dict]) -> str:
        """Encontra a engine mais ociosa apenas dentro de uma lista específica."""
        if not subset:
            # Assumindo que ALL_BACKENDS está disponível no escopo global
            return random.choice(ALL_BACKENDS)
            
        min_load = float("inf")
        candidates = []
        
        for engine in subset:
            m = metrics.get(engine, {})
            if getattr(self, 'threshold_metric', 'total') == "total":
                load = float(m.get("pt_running", 0)) # <-- Mudança aqui
            else:
                val = m.get(self.threshold_metric)
                try:
                    load = float(val) if val is not None else float("inf")
                except Exception:
                    load = float("inf")

            # Coleta candidatos com a menor carga
            if load < min_load:
                min_load = load
                candidates = [engine]
            elif load == min_load:
                candidates.append(engine)
                
        return random.choice(candidates) if candidates else random.choice(ALL_BACKENDS)


    def reorder_preferred_engines(self, program_id: str, metrics: Dict[str, Dict]):
        """
        Reordena as engines favoritas com base na Proporção de Dominância:
        (Requisições deste programa / Total de requisições na engine).
        """
        from routing.process_table import PROCESS_TABLE
        with PROCESS_TABLE.lock:
            entry = PROCESS_TABLE.table.get(program_id)
            if not entry or not entry.get("preferred_engines"):
                return
            
            preferred = entry["preferred_engines"]
            
            def get_dominance_score(engine_id: str) -> tuple:
                total_reqs = int(metrics.get(engine_id, {}).get("local_active_requests", 0))
                
                if total_reqs == 0:
                    return (0.0, 0)
                
                my_reqs = sum(
                    1 for th in entry.get("threads", {}).values()
                    if th.get("engine_id") == engine_id and th.get("state") in ("running", "waiting")
                )
                
                proportion = my_reqs / total_reqs
                return (proportion, my_reqs)

            # Ordena a lista in-place (maior proporção primeiro)
            preferred.sort(key=get_dominance_score, reverse=True)

    async def on_program_complete(self, program_id: str):
        PROCESS_TABLE.clear_preferred_engines(program_id)

class OrderedDynamicAutellixStrategy(BaseDynamicAutellixStrategy):
    """
    Cascata (Ordered): Tenta as engines na ordem em que foram adicionadas.
    Prioriza maximizar o hit rate do Prefix Cache na engine principal.
    """
    async def select_engine(
        self, program_id: Optional[str], num_input_tokens: int, metrics: Dict[str, Dict]
    ) -> str:
        if num_input_tokens <= self.short_request_threshold:
            return self._select_least_used_global(metrics)

        if program_id:
            from routing.process_table import PROCESS_TABLE
            proc = PROCESS_TABLE.get_process(program_id)
            preferred = proc.get("preferred_engines", []) if proc else []

            if preferred:
                selected_engine = None

                for idx, engine in enumerate(preferred):
                    if not self._is_engine_full(engine, metrics):
                        selected_engine = engine
                        break

                if selected_engine:
                    return selected_engine
                

                new_engine = self._select_least_used_global(metrics)
                PROCESS_TABLE.add_preferred_engine(program_id, new_engine)
                return new_engine

        return self._select_least_used_global(metrics)
    
class OrderedReorderDynamicAutellixStrategy(BaseDynamicAutellixStrategy):
    """
    Cascata (Ordered): Tenta as engines na ordem em que foram adicionadas.
    Prioriza maximizar o hit rate do Prefix Cache na engine principal.
    """
    async def select_engine(
        self, program_id: Optional[str], num_input_tokens: int, metrics: Dict[str, Dict]
    ) -> str:
        if num_input_tokens <= self.short_request_threshold:
            return self._select_least_used_global(metrics)

        if program_id:
            from routing.process_table import PROCESS_TABLE
            proc = PROCESS_TABLE.get_process(program_id)
            preferred = proc.get("preferred_engines", []) if proc else []

            if preferred:
                spillover_occurred = False
                selected_engine = None

                for idx, engine in enumerate(preferred):
                    if not self._is_engine_full(engine, metrics):
                        selected_engine = engine
                        break
                    else:
                        spillover_occurred = True

                if selected_engine:
                    if spillover_occurred:
                        self.reorder_preferred_engines(program_id, metrics)
                    return selected_engine
                

                new_engine = self._select_least_used_global(metrics)
                PROCESS_TABLE.add_preferred_engine(program_id, new_engine)
                return new_engine

        return self._select_least_used_global(metrics)

class LeastLoadDynamicAutellixStrategy(BaseDynamicAutellixStrategy):
    """
    Least Load: Avalia todas as engines favoritas e envia para a mais ociosa.
    Prioriza o balanceamento de carga entre o subset de engines do programa.
    """
    async def select_engine(
        self, program_id: Optional[str], num_input_tokens: int, metrics: Dict[str, Dict]
    ) -> str:
        
        if num_input_tokens <= self.short_request_threshold:
            return self._select_least_used_global(metrics)

        if program_id:
            from routing.process_table import PROCESS_TABLE
            proc = PROCESS_TABLE.get_process(program_id)
            preferred = proc.get("preferred_engines", []) if proc else []

            if preferred:
                # 1. Filtra apenas as engines favoritas que NÃO atingiram o limite de capacidade
                available_preferred = [
                    engine for engine in preferred 
                    if not self._is_engine_full(engine, metrics)
                ]

                # 2. Se houver engines disponíveis, usa o método que já consertamos para pegar a mais ociosa
                if available_preferred:
                    return self._select_least_used_subset(available_preferred, metrics)

                # 3. Spillover: Todas as engines favoritas estão cheias
                # (Aloca uma nova engine no cluster global)
                new_engine = self._select_least_used_global(metrics)
                PROCESS_TABLE.add_preferred_engine(program_id, new_engine)
                return new_engine

        # Fallback caso não tenha program_id
        return self._select_least_used_global(metrics)

class ProbabilisticReorderCascadeAutellixStrategy(BaseDynamicAutellixStrategy):
    """
    Cascata Probabilística (Stateless) com Proteção de Exaustão.
    Avalia as engines favoritas em ordem. Faz spillover estocástico 
    se a carga estiver alta. Se esgotar as favoritas, aloca uma nova.
    Se o cluster inteiro esgotar, enfileira na engine menos carregada.
    """
    def __init__(
        self, 
        short_request_threshold: int = 2048, 
        l_min_ratio: float = 0.50,  # 50% da capacidade = começa a vazar
        l_max_ratio: float = 0.95   # 95% da capacidade = vaza 100% das requisições
    ):
        super().__init__(short_request_threshold)
        self.l_min_ratio = l_min_ratio
        self.l_max_ratio = l_max_ratio

    async def select_engine(
        self, program_id: Optional[str], num_input_tokens: int, metrics: Dict[str, Dict]
    ) -> str:
        
        # SHORT REQUEST → Load balance global
        if num_input_tokens <= self.short_request_threshold:
            return self._select_least_used_global(metrics)

        # LONG REQUEST → Cascata Probabilística
        if program_id:
            from routing.process_table import PROCESS_TABLE
            proc = PROCESS_TABLE.get_process(program_id)
            preferred = proc.get("preferred_engines", []) if proc else []

            if preferred:
                spillover_occurred = False
                selected_engine = None

                for engine in preferred:
                    capacity = BACKEND_PARALLELISM.get(engine, 10)
                    
                    # CORREÇÃO: Usar a mesma lógica dinâmica de métricas da classe base
                    m = metrics.get(engine, {})
                    if getattr(self, 'threshold_metric', 'total') == "total":
                        load = float(m.get("pt_running", 0)) # <-- Mudança aqui
                    else:
                        val = m.get(self.threshold_metric)
                        try:
                            load = float(val) if val is not None else 0.0
                        except Exception:
                            load = 0.0
                    
                    l_min = capacity * self.l_min_ratio
                    l_max = capacity * self.l_max_ratio

                    # 1. Carga confortável: Fica nesta engine
                    if load <= l_min:
                        selected_engine = engine
                        break
                        
                    # 2. Carga crítica: Spillover garantido
                    if load >= l_max:
                        spillover_occurred = True
                        continue  # CORREÇÃO: Pula imediatamente para a próxima engine da lista
                        
                    # 3. Zona de transição: Probabilidade de spillover
                    p_spillover = (load - l_min) / (l_max - l_min)
                    if random.random() > p_spillover:
                        selected_engine = engine
                        break
                    else:
                        spillover_occurred = True

                if selected_engine:
                    if spillover_occurred:
                        # Assumindo que o método reorder_preferred_engines existe
                        self.reorder_preferred_engines(program_id, metrics)
                    return selected_engine

                # --- TRATAMENTO DE EXAUSTÃO ---
                
                # Se o loop terminou sem 'selected_engine', todas as favoritas deram spillover.
                # Precisamos de uma engine nova. Quais ainda não foram usadas por este programa?
                available_new_engines = [e for e in ALL_BACKENDS if e not in preferred]

                if not available_new_engines:
                    # Esgotou o cluster inteiro
                    selected_engine = self._select_least_used_subset(preferred, metrics)
                    self.reorder_preferred_engines(program_id, metrics)
                    return selected_engine
                
                # SCALE-OUT: Aloca a engine mais ociosa dentre as que ainda não são favoritas
                new_engine = self._select_least_used_subset(available_new_engines, metrics)
                PROCESS_TABLE.add_preferred_engine(program_id, new_engine)
                return new_engine

        # Fallback de segurança global
        return self._select_least_used_global(metrics)

class ProbabilisticCascadeAutellixStrategy(BaseDynamicAutellixStrategy):
    """
    Cascata Probabilística (Stateless) com Proteção de Exaustão.
    Avalia as engines favoritas em ordem. Faz spillover estocástico 
    se a carga estiver alta. Se esgotar as favoritas, aloca uma nova.
    Se o cluster inteiro esgotar, enfileira na engine menos carregada.
    """
    def __init__(
        self, 
        short_request_threshold: int = 2048, 
        l_min_ratio: float = 0.50,  # 50% da capacidade = começa a vazar
        l_max_ratio: float = 0.95   # 95% da capacidade = vaza 100% das requisições
    ):
        super().__init__(short_request_threshold)
        self.l_min_ratio = l_min_ratio
        self.l_max_ratio = l_max_ratio

    async def select_engine(
        self, program_id: Optional[str], num_input_tokens: int, metrics: Dict[str, Dict]
    ) -> str:
        
        # SHORT REQUEST → Load balance global
        if num_input_tokens <= self.short_request_threshold:
            return self._select_least_used_global(metrics)

        # LONG REQUEST → Cascata Probabilística
        if program_id:
            from routing.process_table import PROCESS_TABLE
            proc = PROCESS_TABLE.get_process(program_id)
            preferred = proc.get("preferred_engines", []) if proc else []

            if preferred:
                selected_engine = None

                for engine in preferred:
                    capacity = BACKEND_PARALLELISM.get(engine, 10)
                    
                    # CORREÇÃO: Usar a mesma lógica dinâmica de métricas da classe base
                    m = metrics.get(engine, {})
                    if getattr(self, 'threshold_metric', 'total') == "total":
                        load = float(m.get("pt_running", 0)) # <-- Mudança aqui
                    else:
                        val = m.get(self.threshold_metric)
                        try:
                            load = float(val) if val is not None else 0.0
                        except Exception:
                            load = 0.0
                    
                    l_min = capacity * self.l_min_ratio
                    l_max = capacity * self.l_max_ratio

                    # 1. Carga confortável: Fica nesta engine
                    if load <= l_min:
                        selected_engine = engine
                        break
                        
                    # 2. Carga crítica: Spillover garantido
                    if load >= l_max:
                        continue  # CORREÇÃO: Pula imediatamente para a próxima engine da lista
                        
                    # 3. Zona de transição: Probabilidade de spillover
                    p_spillover = (load - l_min) / (l_max - l_min)
                    if random.random() > p_spillover:
                        selected_engine = engine
                        break

                if selected_engine:
                    return selected_engine

                # --- TRATAMENTO DE EXAUSTÃO ---
                
                # Se o loop terminou sem 'selected_engine', todas as favoritas deram spillover.
                # Precisamos de uma engine nova. Quais ainda não foram usadas por este programa?
                available_new_engines = [e for e in ALL_BACKENDS if e not in preferred]

                if not available_new_engines:
                    # Esgotou o cluster inteiro
                    selected_engine = self._select_least_used_subset(preferred, metrics)
                    return selected_engine
                
                # SCALE-OUT: Aloca a engine mais ociosa dentre as que ainda não são favoritas
                new_engine = self._select_least_used_subset(available_new_engines, metrics)
                PROCESS_TABLE.add_preferred_engine(program_id, new_engine)
                return new_engine

        # Fallback de segurança global
        return self._select_least_used_global(metrics)




class LoadBalancer:
    """Coordinator: metrics collection, logging and strategy dispatch."""

    def __init__(self):
        self.strategy: LoadBalancerStrategy = self._init_strategy()
        self.metrics_cache: Dict[str, Dict] = {engine: {} for engine in ALL_BACKENDS}
        self.metrics_lock = Lock()
        self.last_metrics_log = 0.0
        self.metrics_query_task: Optional[asyncio.Task] = None
        self.load_history = deque()
        self.admission_lock = Lock()
        self.last_admission_time = 0.0
        self.has_received_traffic = False

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

        if name == "ordered-dynamic-autellix":
            return OrderedDynamicAutellixStrategy(LOAD_BALANCER_SHORT_REQUEST_THRESHOLD)

        if name == "least-load-dynamic-autellix":
            return LeastLoadDynamicAutellixStrategy(LOAD_BALANCER_SHORT_REQUEST_THRESHOLD)
        
        if name == "probabilistic-cascade-autellix":
            return ProbabilisticCascadeAutellixStrategy(
                short_request_threshold=LOAD_BALANCER_SHORT_REQUEST_THRESHOLD
            )
        
        if name == "ordered-dynamic-autellix-reorder":
            return OrderedReorderDynamicAutellixStrategy(LOAD_BALANCER_SHORT_REQUEST_THRESHOLD)
        
        if name == "probabilistic-cascade-autellix-reorder":
            return ProbabilisticReorderCascadeAutellixStrategy(
                short_request_threshold=LOAD_BALANCER_SHORT_REQUEST_THRESHOLD
            )
        
        # ------------------------
        # Safe Default
        # ------------------------
        return AutelixStrategy(LOAD_BALANCER_SHORT_REQUEST_THRESHOLD)


      
    async def select_engine(self, program_id: Optional[str], num_input_tokens: int) -> str: 
        metrics = self._get_cached_metrics()
        
        # INJEÇÃO EM TEMPO REAL: Pega as requisições em execução no exato milissegundo
        if hasattr(PROCESS_TABLE, "get_engine_running_counts"):
            running_counts = PROCESS_TABLE.get_engine_running_counts()
            for engine in ALL_BACKENDS:
                metrics.setdefault(engine, {})
                metrics[engine]["pt_running"] = running_counts.get(engine, 0)

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
        
        total_running_cluster = 0 # Guarda o total deste ciclo
        
        with self.metrics_lock:
            for engine, res in zip(ALL_BACKENDS, results):
                if isinstance(res, dict):
                    self.metrics_cache.setdefault(engine, {})
                    self.metrics_cache[engine].update(res)
                else:
                    self.metrics_cache.setdefault(engine, {})["query_error"] = str(res)

            if hasattr(PROCESS_TABLE, "get_all_engine_workloads"):
                workloads = PROCESS_TABLE.get_all_engine_workloads()
            else:
                workloads = {}
                
            running_counts = PROCESS_TABLE.get_engine_running_counts() if hasattr(PROCESS_TABLE, "get_engine_running_counts") else {}

            for engine in ALL_BACKENDS:
                self.metrics_cache.setdefault(engine, {})
                local_active = workloads.get(engine, 0)
                self.metrics_cache[engine]["local_active_requests"] = local_active
                
                # Agora sim, o pt_running existe e é preciso
                pt_running_val = running_counts.get(engine, 0)
                total_running_cluster += int(pt_running_val)
                
        window_s = getattr(settings, 'routing/load_balancer.pyON_WINDOW_S', 30.0)
        now = time.time()
        
        with self.admission_lock:
            # Vira a chave na primeira vez que a ocupação for maior que zero
            if total_running_cluster > 0:
                self.has_received_traffic = True
                
            # Só grava no histórico se a ignição já foi dada
            if self.has_received_traffic:
                self.load_history.append((now, total_running_cluster))
                
            # Remove amostras antigas que saíram da janela
            while self.load_history and now - self.load_history[0][0] > window_s:
                self.load_history.popleft()

    # --- NOVO: Método de verificação da média móvel ---
    def can_admit_new_program(self) -> bool:
        """Calcula a média móvel e decide se o cluster tem capacidade para novos programas."""
        # Feature Toggle: Se estiver desligado, permite passar direto
        if not getattr(settings, 'ENABLE_ADMISSION_CONTROL', False):
            self.last_admission_time = time.time()
            return True
        
        now = time.time()
        cooldown_s = getattr(settings, 'ADMISSION_COOLDOWN_S', 1.0)
            
        with self.admission_lock:
            if now - self.last_admission_time < cooldown_s:
                return False
            if not self.load_history:
                self.last_admission_time = time.time()
                return True
                
            # Calcula a média aritmética do período
            total_load = sum(count for _, count in self.load_history)
            avg_load = total_load / len(self.load_history)
            
        # Capacidade máxima física do cluster inteiro
        max_capacity = 192 #sum(BACKEND_PARALLELISM.get(e, 1) for e in ALL_BACKENDS)
        threshold_pct = getattr(settings, 'ADMISSION_THRESHOLD_PCT', 0.8)
        
        # Só admite se a média móvel de ocupação for menor que o limiar
        # print(f"Admission Control: avg_load={avg_load:.2f}, max_capacity={max_capacity}, threshold_pct={threshold_pct}")
        if (avg_load < (max_capacity * threshold_pct)):
            self.last_admission_time = time.time()
            return True
        else:
            return False

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
 
            def first_not_none(*values):
                for v in values:
                    if v is not None:
                        return v
                return None
 
            running = first_not_none(em.get("running"), em.get("local_active_requests"), em.get("active"), 0)
            waiting = first_not_none(em.get("queue_size"), em.get("waiting"), 0)
 
            kv = first_not_none(em.get("kv_cache_percent"), find_metric(["kv", "cache"]))
            prefix_hit = first_not_none(em.get("prefix_cache_hit_rate"), find_metric(["prefix", "hit"]))

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