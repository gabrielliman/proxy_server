import time
from fastapi import APIRouter, Request, HTTPException
import asyncio
from uuid import uuid4
import os
import sys
from config.settings import (
    MODEL_ROUTES,
    DISPATCH_MODE,
    MODEL,
)

from routing.load_balancer import LOAD_BALANCER
from routing.process_table import PROCESS_TABLE

# Inicialização das filas (mantido conforme seu original)
from routing.queue_manager import init_queues
init_queues()

dispatcher = None

from utils.tokenizer_utils import get_tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in text using tokenizer."""
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))

router = APIRouter()

def get_dispatcher():
    global dispatcher
    if dispatcher is not None:
        return dispatcher

    if DISPATCH_MODE == "direct":
        from routing.dispatcher.direct import DirectDispatcher
        dispatcher = DirectDispatcher()
    elif DISPATCH_MODE == "worker_pool":
        from routing.dispatcher.worker import WorkerPoolDispatcher
        dispatcher = WorkerPoolDispatcher()
    elif DISPATCH_MODE == "semaphore":
        from routing.dispatcher.semaphore import SemaphoreDispatcher
        dispatcher = SemaphoreDispatcher()
    else:
        raise RuntimeError(f"Invalid DISPATCH_MODE: {DISPATCH_MODE}")

    return dispatcher


@router.post("/v1/chat/completions")
async def chat_completion(request: Request):
    program_id = "cuad-experiment" 
    data = await request.json()
    model = data.get("model")
    
    from config.settings import MODEL_ROUTES
    if not model:
        raise HTTPException(400, "Campo 'model' é obrigatório")
    
    candidates = MODEL_ROUTES.get(model)
    if not candidates:
        raise HTTPException(400, f"Modelo {model} não configurado.")

    # -----------------------------
    # Estimate input tokens (using tokenizer for accurate count)
    # -----------------------------
    messages = data.get("messages", [])
    # Combine all message content for token counting
    input_text = "\n".join(m.get("content", "") for m in messages if isinstance(m, dict))
    num_input_tokens = count_tokens(input_text)

    # -----------------------------
    # Select engine via Autellix LB
    # -----------------------------
    backend = None
    while backend is None:
        backend = await LOAD_BALANCER.select_engine(
            program_id=program_id,
            num_input_tokens=num_input_tokens,
            candidates=candidates 
        )
        if backend is None:
            await asyncio.sleep(0.05)

    # -----------------------------
    # Instrument arrival (with prefill tokens for KV metric)
    # -----------------------------
    call_id = str(uuid4())
    PROCESS_TABLE.record_call_arrival(program_id, call_id, prefill_tokens=num_input_tokens)

    # -----------------------------
    # Dispatch
    # -----------------------------
    disp = get_dispatcher()

    return await disp.dispatch(
        backend=backend,
        data=data,
        program_id=program_id,
        num_input_tokens=num_input_tokens,
        candidates=candidates 
    )
    print(f"🎯 [ENDPOINTS CHAT] Engine selecionada: {backend}")
    
    if not backend:
        raise HTTPException(503, "Nenhum backend disponível para este modelo.")

    call_id = str(uuid4())
    print(f"🔍 [ENDPOINTS PID CHECK - ENDPOINT] Requisição iniciada. Processo: {os.getpid()} | Call ID: {call_id}")
    PROCESS_TABLE.record_call_arrival(program_id, call_id)
    disp = get_dispatcher()    
    use_tie = data.get("use_tie", False)

    # Antes de criar a dispatch_task
    payload_size = sys.getsizeof(str(data)) / 1024  # em KB
    print(f"📡 [DEBUG - PAYLOAD] Tamanho do JSON: {payload_size:.2f} KB")
    start_dispatch = time.time()

    
    dispatch_task = asyncio.create_task(
        disp.dispatch(backend=backend, data=data, program_id=program_id, call_id=call_id)
    )
    print(f"🚀 [DEBUG - DISPATCH] Task criada. Tempo de overhead: {time.time() - start_dispatch:.4f}s")
        
    if use_tie:
        # --- NOVO: Incrementa ANTES de checar ---
        call_count = PROCESS_TABLE.increment_call_count(program_id)
        
        print(f"📊 [DEBUG] Call: {call_id} | Program: {program_id} | Novo Count: {call_count}")

        # Agora a checagem usará o valor atualizado
        if call_count <= 15:
            print(f"🧬 [WARM-UP] Deixando passar Call {call_id} (Rodada {call_count}/15)")
            res = await dispatch_task
            print(f"✅ [WARM-UP] Resposta recebida para {call_id}!")
            return res

        # --------------------------------------------
        
        try:
            
            # monitor_start = time.time()
            # Criamos uma tarefa leve para monitorar a desconexão sem travar o loop
            async def watch_disconnection():
                try:
                    while not await request.is_disconnected():
                        # Polling muito curto, apenas 100ms para ser reativo
                        await asyncio.sleep(0.1)
                    return "disconnected"
                except Exception:
                    return "disconnected"
                
            # Criamos o watcher como tarefa
            disconnection_task = asyncio.create_task(watch_disconnection())

            done, pending = await asyncio.wait(
                [dispatch_task, disconnection_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Se o cliente desconectou primeiro
            if disconnection_task in done:
                 # Descobrir POR QUE a desconexão venceu
                reason = disconnection_task.result()
                print(f"🔍 [DEBUG - RACE] Poda venceu! Motivo: {reason} | Task pendente: {pending}")
                
                print(f"✂️ [ENDPOINTS PID CHECK - PODA] Tentando podar no Processo: {os.getpid()} | Call ID: {call_id}")
                print(f"✂️ [ENDPOINTS CHAT] PODA DETECTADA! Call ID: {call_id}")
                
                # 1. Cancela a tarefa de despacho (isso mata a espera pelo Future)
                dispatch_task.cancel()

                # 2. Tenta cancelar o Worker (caso ele já tenha começado)
                worker_task = PROCESS_TABLE.get_active_task(call_id)
                if worker_task:
                    print(f"🎯 [DEBUG] Worker ativo cancelado para {call_id}")
                    worker_task.cancel()
                else:
                    # 3. Se não achou o worker, marcamos na PROCESS_TABLE que essa Call_ID foi abortada
                    # O Worker deve checar isso antes de começar a processar!
                    print(f"📥 [DEBUG] Marcando Call {call_id} como ABORTADA na Fila.")
                    PROCESS_TABLE.record_call_error(program_id, call_id, "TIE_PRUNED_IN_QUEUE")

                # return {"error": "TIE_PRUNED"}
                model_used = data.get("model", "unknown")
                # No endpoints_chat.py, dentro do bloco 'if disconnection_task in done'
                import json
                mock_content = json.dumps({"results": [], "metadata": {"pruned": True}})
                return {
                    "choices": [{"message": {"content": mock_content, "role": "assistant"}, "finish_reason": "stop"}],
                    "model": model_used,
                    "usage": {"total_tokens": 0}
                }
                            
            elif dispatch_task in done:
                print(f"✅ [DEBUG - RACE] GPU venceu! Retornando dados reais.")
 # Se o dispatch_task terminou, cancelamos o watcher e retornamos o resultado
            disconnection_task.cancel()
            return dispatch_task.result() # Use .result() pois ela já está no conjunto 'done'

        except Exception as e:
            print(f"❌ Erro no fluxo TIE: {e}")
            return {"error": str(e)}
    else:
        return await dispatch_task
    
