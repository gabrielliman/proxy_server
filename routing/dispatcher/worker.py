# routing/dispatcher/workers.py

import asyncio
import httpx
import time
from routing.dispatcher.base import BaseDispatcher
from routing.queue_manager import get_queue, put_request, get_request
from config.settings import REQUEST_TIMEOUT, BACKEND_PARALLELISM, ALL_BACKENDS, SCHEDULER, MODEL
from utils.tokenizer_utils import get_tokenizer
from routing.load_balancer import LOAD_BALANCER


class WorkerPoolDispatcher(BaseDispatcher):
    workers_started = False
    shared_client = None
    engine_semaphores = {}  # <-- CAMADA 2: Dicionário para guardar os limites físicos (Semáforos)

    async def start_workers(self):
        if WorkerPoolDispatcher.workers_started:
            return
        WorkerPoolDispatcher.workers_started = True

        loop = asyncio.get_running_loop()

        timeout_config = httpx.Timeout(
            connect=30.0,      
            read=300.0,        
            write=30.0,        
            pool=30.0
        )

        limits = httpx.Limits(
            max_keepalive_connections=500, 
            max_connections=2000,
            keepalive_expiry=30.0
        )

        WorkerPoolDispatcher.shared_client = httpx.AsyncClient(
            timeout=timeout_config, 
            limits=limits
        )

        # Inicializa os Semáforos com o limite exato de capacidade de cada backend
        for backend in ALL_BACKENDS:
            capacity = BACKEND_PARALLELISM.get(backend, 1)
            WorkerPoolDispatcher.engine_semaphores[backend] = asyncio.Semaphore(capacity)

        global_queue_name = "global_cluster"
        queue = get_queue(global_queue_name)
        
        # 1. Calcula o total de workers somando a capacidade de todas as engines
        total_workers = 0
        for backend in ALL_BACKENDS:
            total_workers += BACKEND_PARALLELISM.get(backend, 1)

        # 2. Inicia o pool global de workers apontando para a fila unificada
        for worker_id in range(total_workers):
            loop.create_task(self.worker_loop(global_queue_name, queue, worker_id, WorkerPoolDispatcher.shared_client))


    async def worker_loop(self, queue_name, queue, worker_id, client):
        from routing.process_table import PROCESS_TABLE

        while True:
            # Puxa da Fila Global
            if SCHEDULER in ("plas", "atlas"):
                item = await get_request(queue_name)
            else:
                item = await queue.get()
            
            data = item["data"]
            future = item["future"]
            pid = item.get("program_id")
            cid = item.get("call_id")
            input_tokens = item.get("input_tokens", 0) 

            if pid and cid:
                PROCESS_TABLE.record_dequeue(pid, cid)

            start = None
            backend = None
            try:
                start = time.time()
                output_tokens = None
                
                # 3. JUST-IN-TIME ROUTING
                backend = await LOAD_BALANCER.select_engine(pid, input_tokens)

                # --- CAMADA 1: FEEDBACK INSTANTÂNEO PARA O LB ---
                # Atualiza a métrica na hora. Se 30 workers acordarem juntos,
                # o LB vai rotear perfeitamente porque a métrica "running" sobe em tempo real.
                with LOAD_BALANCER.metrics_lock:
                    if backend in LOAD_BALANCER.metrics_cache:
                        LOAD_BALANCER.metrics_cache[backend]["running"] = LOAD_BALANCER.metrics_cache[backend].get("running", 0) + 1

                # --- CAMADA 2: BARREIRA FÍSICA (HARD LIMIT) ---
                # Garante que mesmo sob pressão, a engine nunca excederá o BACKEND_PARALLELISM
                sem = WorkerPoolDispatcher.engine_semaphores[backend]
                
                async with sem:
                    if pid and cid:
                        # O tempo de execução real só começa quando passa do semáforo
                        PROCESS_TABLE.record_call_start(pid, cid, engine_id=backend, start_time=time.time())

                    resp = await client.post(f"{backend}/v1/chat/completions", json=data)
                    resp_json = resp.json()
                    
                    if resp_json and "choices" in resp_json:
                        choice = resp_json["choices"][0] if resp_json["choices"] else {}
                        text = choice.get("message", {}).get("content", "") or choice.get("text", "")
                        if text:
                            tok = get_tokenizer()
                            output_tokens = len(tok.encode(text, add_special_tokens=False))

                if not future.done():
                    future.set_result(resp_json)

            except Exception as e:
                print(f"[WORKER ERROR] {e}")
                if not future.done():
                    future.set_exception(e)

            finally:
                end = time.time()
                if pid and cid:
                    PROCESS_TABLE.record_call_completion(pid, cid, completion_time=end, output_tokens=output_tokens)
                
                # Libera o espaço na métrica instantânea do Load Balancer
                if backend:
                    with LOAD_BALANCER.metrics_lock:
                        if backend in LOAD_BALANCER.metrics_cache:
                            curr = LOAD_BALANCER.metrics_cache[backend].get("running", 1)
                            LOAD_BALANCER.metrics_cache[backend]["running"] = max(0, curr - 1)

                if hasattr(queue, "task_done") and SCHEDULER not in ("plas", "atlas"):
                    queue.task_done()

    # Mantivemos 'backend' na assinatura para não quebrar quem chama o dispatch (ex: o servidor principal)
    async def dispatch(self, backend: str, data: dict, program_id: str = None, call_id: str = None, input_tokens: int = 0) -> dict:
        from routing.process_table import PROCESS_TABLE
        await self.start_workers()

        global_queue_name = "global_cluster"
        queue = get_queue(global_queue_name)
        loop = asyncio.get_running_loop()

        future = loop.create_future()
        
        if program_id and call_id:
            PROCESS_TABLE.record_enqueue(program_id, call_id)

        item = {
            "data": data, 
            "future": future, 
            "program_id": program_id, 
            "call_id": call_id,
            "input_tokens": input_tokens 
        }
        
        if SCHEDULER in ("plas", "atlas"):
            from config.settings import ANTI_STARVATION_RATIO_THRESHOLD
            if program_id and PROCESS_TABLE.should_promote_to_q1(program_id, ANTI_STARVATION_RATIO_THRESHOLD):
                PROCESS_TABLE.reset_wait_and_service_for_promotion(program_id)
                await put_request(global_queue_name, 0.0, item)
            else:
                if SCHEDULER == "atlas":
                    from routing.scheduler_atlas import compute_priority
                else:
                    from routing.scheduler_plas import compute_priority
                    
                priority, seq = compute_priority(program_id, call_id)
                await put_request(global_queue_name, priority, item)
        else:
            await queue.put(item)

        return await future