import asyncio
import httpx
import time
import json
from routing.dispatcher.base import BaseDispatcher
from routing.queue_manager import get_queue, put_request, get_request
from config.settings import REQUEST_TIMEOUT, BACKEND_PARALLELISM, ALL_BACKENDS, SCHEDULER, MODEL
from routing.process_table import PROCESS_TABLE
from routing.scheduler_plas import compute_priority
import os

class WorkerPoolDispatcher(BaseDispatcher):
    workers_started = False
    
    def _reconstruct_json(self, text, model_name):
        """Converte o texto acumulado do stream no formato JSON esperado pelo Palimpzest."""
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0, # O vLLM envia isso no último chunk se configurado
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    _active_worker_tasks = []


    async def start_workers(self):
        # Use apenas a variável da classe para evitar confusão
        if WorkerPoolDispatcher.workers_started:
            print("👷 [WORKER] Workers já estavam rodando.")
            return
            
        WorkerPoolDispatcher.workers_started = True
        print(f"👷 [WORKER] Iniciando {sum(BACKEND_PARALLELISM.values())} Workers...")

        for backend in ALL_BACKENDS:
            queue = get_queue(backend)
            n_workers = BACKEND_PARALLELISM.get(backend, 1)
            for worker_id in range(n_workers):
                # Cria a tarefa UMA VEZ e guarda a referência
                task = asyncio.create_task(self.worker_loop(backend, queue, worker_id))
                self._active_worker_tasks.append(task)
                
                
                
    async def worker_loop(self, backend, queue, worker_id):
        print(f"🐝 [WORKER {worker_id}] Pronto e aguardando tarefa...")
        while True:
            try:
                print(f"🕵️ [WORKER {worker_id}] Olhando para a fila: {backend}")
                if SCHEDULER == "plas":
                    item = await get_request(backend) 
                else:
                    item = await queue.get()
                print(f"📦 [WORKER {worker_id}] PEGOU TAREFA! Call ID: {item.get('call_id')}")
                data = item["data"].copy() 
                future = item["future"]
                pid, cid = item.get("program_id"), item.get("call_id")
                # No worker_loop, antes de entrar no use_tie:
                # VERIFICAÇÃO DA FLAG: Se não for passado, o default é FALSE (Abacus Original)
                use_tie = data.get("use_tie", False)
                
                # --- CAMINHO A: TIE ATIVADO (CUAD / BIODEX) ---
                if use_tie:
                    # --- CHECK DE PODA PRECOCE ---
                    if PROCESS_TABLE.is_call_pruned(pid, cid): 
                        future.set_exception(asyncio.CancelledError())
                        continue
                    # -----------------------------
                    # async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT,
                        # limits=httpx.Limits(max_connections=1, max_keepalive_connections=0)) as client:
                    async with httpx.AsyncClient(
                        timeout=600.0, # Aumente para 10 minutos (41 categorias pesam muito)
                        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5) # Seja menos restritivo
                    ) as client:
                        
                        data["stream"] = True 
                        current_task = asyncio.current_task()
                        if cid:
                            PROCESS_TABLE.register_active_task(cid, current_task) 
                        
                        full_response_text = ""
                        try:
                            if pid and cid:
                                PROCESS_TABLE.record_call_start(pid, cid, backend, time.time())
                            
                            stream_start = time.time()    
                            print(f"🔗 [DEBUG - WORKER] Tentando abrir conexão stream com vLLM: {backend}")
                            
                            async with client.stream("POST", f"{backend}/v1/chat/completions",
                                                    json=data, headers={"Connection": "close"}) as resp:
                                # --- INSIRA ESTE BLOCO AQUI ---
                                ttfb = time.time() - stream_start # Calcula o tempo até o primeiro sinal de vida
                                print(f"⏱️ [DEBUG - WORKER] Resposta inicial (Headers) de {cid} após {ttfb:.2f}s | Status: {resp.status_code}")
                                
                                if resp.status_code != 200:
                                    error_detail = await resp.aread()
                                    print(f"❌ [DEBUG - WORKER] vLLM rejeitou com erro {resp.status_code}: {error_detail}")
                                # ------------------------------
                                try:
                                    async for line in resp.aiter_lines():
                                        if line.startswith("data: "):
                                            payload = line[6:].strip()
                                            if payload == "[DONE]":
                                                break
                                            try:
                                                chunk = json.loads(payload)
                                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                                    content = chunk["choices"][0].get("delta", {}).get("content", "")
                                                    full_response_text += content
                                            except:
                                                continue
                                except asyncio.TimeoutError:
                                    print(f"⚠️ Timeout no stream do backend {backend}")
                            
                            final_json = self._reconstruct_json(full_response_text, data.get("model"))
                            if not future.done():
                                future.set_result(final_json)

                        except asyncio.CancelledError:
                            if full_response_text.strip():
                                final_json = self._reconstruct_json(full_response_text, data.get("model"))
                                if not future.done():
                                    future.set_result(final_json)
                            else:
                                if not future.done():
                                    future.set_exception(asyncio.CancelledError())
                            
                            if pid and cid:
                                PROCESS_TABLE.record_call_error(pid, cid, "TIE_PRUNED")
                            raise
                        finally:
                            if cid:
                                PROCESS_TABLE.unregister_active_task(cid)

                # --- CAMINHO B: ABACUS ORIGINAL COM PODA (PRUNING) ---
                else:
                    # 1. LOGICA DE PODA: Decidir se vamos executar ou "fingir"
                    # Aqui você define o seu critério (ex: uma flag no data ou metadados do processo)
                    should_prune = data.get("use_tie", False) # Exemplo de flag

                    if should_prune:
                        # Retornamos um JSON fake que o Palimpzest entenda
                        mock_response = self._reconstruct_json(
                            text="[PLAN_PRUNED_BY_TIE_PROXY]", 
                            model_name=data.get("model", "pruned-model")
                        )
                        if not future.done():
                            future.set_result(mock_response)
                        
                        # Registramos como erro ou skip na tabela de processos para sua análise posterior
                        if pid and cid:
                            PROCESS_TABLE.record_call_error(pid, cid, "PRUNED_BY_PROXY")
                    
                    else:
                        # Execução Normal (Sem Poda)
                        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                            if pid and cid:
                                PROCESS_TABLE.record_call_start(pid, cid, backend, time.time())
                            
                            # PROTEÇÃO: Garante que o JSON não está malformado (evita o 400)
                            if not data.get("messages") or not data.get("model"):
                                raise ValueError(f"JSON malformado detectado para {backend}: {data}")

                            response = await client.post(f"{backend}/v1/chat/completions", json=data)
                            
                            # Se der 400 aqui, o print abaixo vai te mostrar EXATAMENTE o que o vLLM cuspiu
                            if response.status_code == 400:
                                print(f"❌ [VLLM REJECTED] Detalhes do 400: {response.text}")
                            
                            response.raise_for_status()
                            
                            if not future.done():
                                future.set_result(response.json())
                                

            except Exception as e:
                print(f"❌ [WORKER ERROR] {backend}: {e}")
                if 'future' in locals() and not future.done():
                    future.set_exception(e)
                await asyncio.sleep(1) 

            finally:
                if 'pid' in locals() and 'cid' in locals() and pid and cid:
                    PROCESS_TABLE.record_call_completion(pid, cid, completion_time=time.time())
                if SCHEDULER != "plas" and 'queue' in locals() and hasattr(queue, "task_done"):
                    queue.task_done()        

    async def dispatch(self, backend: str, data: dict, program_id: str = None, call_id: str = None) -> dict:
        from routing.process_table import PROCESS_TABLE
        await self.start_workers()

        queue = get_queue(backend)
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        if program_id and call_id:
            PROCESS_TABLE.record_enqueue(program_id, call_id)

        # Passamos uma cópia profunda para garantir isolamento total
        item = {"data": data.copy(), "future": future, "program_id": program_id, "call_id": call_id}
        
        if SCHEDULER == "plas":
            from config.settings import ANTI_STARVATION_RATIO_THRESHOLD
            if program_id and PROCESS_TABLE.should_promote_to_q1(program_id, ANTI_STARVATION_RATIO_THRESHOLD):
                PROCESS_TABLE.reset_wait_and_service_for_promotion(program_id)
                await put_request(backend, 0.0, item)
            else:
                priority, seq = compute_priority(program_id, call_id)
                await put_request(backend, priority, item)
        else:
            await queue.put(item)

        return await future
