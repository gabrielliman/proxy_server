# from fastapi import FastAPI
# import uvicorn

# from api.endpoints_chat import router as chat_router
# from api.endpoints_completion import router as completion_router
# from api.endpoints_status import router as status_router
# from metrics.monitor import start_monitoring_tasks
# from routing.process_table import PROCESS_TABLE
# from routing.load_balancer import LOAD_BALANCER
# from config.settings import PROCESS_TABLE_PRUNE_TTL, PROCESS_TABLE_PRUNE_INTERVAL
# # No topo do seu main.py, importe a função que acabamos de criar
# from routing.queue_manager import clear_all_queues # ajuste o caminho se necessário

# app = FastAPI()

# @app.on_event("startup")
# async def startup_event():
#     from config.settings import MODEL_ROUTES # Importe aqui para garantir o reload
#     print(f"\n[DEBUG PROXY] Rotas de modelos carregadas: {MODEL_ROUTES}")
#     start_monitoring_tasks(app)
#     # start process table pruner
#     PROCESS_TABLE.start_async_pruner(PROCESS_TABLE_PRUNE_INTERVAL, PROCESS_TABLE_PRUNE_TTL)
#     # start load balancer metrics collector (logs engine metrics periodically)
#     try:
#         await LOAD_BALANCER.start_metrics_collector()
#     except Exception:
#         # don't fail startup if collector can't start
#         pass


# @app.on_event("shutdown")
# async def shutdown_event():
#     PROCESS_TABLE.stop_async_pruner()
#     try:
#         await LOAD_BALANCER.stop_metrics_collector()
#     except Exception:
#         pass

# # nova função
# @app.post("/reset_queues")
# async def reset_queues():
#     """Endpoint para zerar o experimento e limpar as filas."""
#     try:
#         removed = clear_all_queues()
#         print(f"🧹 [MAIN - PROXY] {removed} requisições removidas das filas.")
#         return {"status": "ok", "cleared_count": removed}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}
    

# app.include_router(chat_router)
# app.include_router(completion_router)
# app.include_router(status_router)


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8080)



from fastapi import FastAPI
import uvicorn
import asyncio

from api.endpoints_chat import router as chat_router
from api.endpoints_completion import router as completion_router
from api.endpoints_status import router as status_router
from metrics.monitor import start_monitoring_tasks
from routing.process_table import PROCESS_TABLE
from routing.load_balancer import LOAD_BALANCER
from config.settings import PROCESS_TABLE_PRUNE_TTL, PROCESS_TABLE_PRUNE_INTERVAL
from routing.queue_manager import clear_all_queues 

app = FastAPI()

async def perform_full_system_reset(cancel_async_tasks: bool = False):
    """
    Versão corrigida: Limpa o estado lógico sem quebrar o servidor.
    """
    print("\n🧹 [SYSTEM] Iniciando limpeza de estado...")
    
    # 1. Limpa as filas (Crucial para o experimento)
    try:
        removed_queues = clear_all_queues()
        print(f"📦 [RESET] {removed_queues} requisições removidas das filas.")
    except Exception as e:
        print(f"⚠️ Erro ao limpar filas: {e}")

    # 2. Reseta a Process Table (Zera call_counts para o Warm-up)
    try:
        # Limpamos os dados, mas mantemos o objeto vivo
        PROCESS_TABLE.active_tasks = {}
        PROCESS_TABLE.program_stats = {}
        print("📋 [RESET] PROCESS_TABLE (warm-up e stats) zerada.")
    except Exception as e:
        print(f"⚠️ Erro ao resetar Process Table: {e}")

    # 3. Cancelamento Opcional (Só usamos via Endpoint, nunca no Startup)
    if cancel_async_tasks:
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        # Filtramos para não matar tarefas do Uvicorn/Lifespan
        count = 0
        for t in tasks:
            # Só cancelamos tarefas que pareçam ser de Workers ou do Dispatcher
            # (Geralmente elas não têm nomes protegidos do sistema)
            if "lifespan" not in str(t).lower() and "server" not in str(t).lower():
                t.cancel()
                count += 1
        print(f"🛑 [RESET] {count} tarefas de processamento interrompidas.")

    print("✨ [SYSTEM] Estado resetado.\n")

@app.on_event("startup")
async def startup_event():
    # NO STARTUP: Limpamos apenas os DADOS (warm-up e filas)
    # cancel_async_tasks=False evita o erro que você viu
    await perform_full_system_reset(cancel_async_tasks=False)
    
    from config.settings import MODEL_ROUTES 
    print(f"[DEBUG PROXY] Rotas de modelos carregadas: {MODEL_ROUTES}")
    
    start_monitoring_tasks(app)
    PROCESS_TABLE.start_async_pruner(PROCESS_TABLE_PRUNE_INTERVAL, PROCESS_TABLE_PRUNE_TTL)
    
    try:
        await LOAD_BALANCER.start_metrics_collector()
    except Exception:
        pass

@app.on_event("shutdown")
async def shutdown_event():
    PROCESS_TABLE.stop_async_pruner()
    try:
        await LOAD_BALANCER.stop_metrics_collector()
    except Exception:
        pass

@app.post("/reset_system")
async def reset_system_endpoint():
    """
    Via API, podemos ser um pouco mais agressivos se necessário.
    """
    try:
        # Aqui podemos tentar cancelar tarefas se o sistema estiver travado
        await perform_full_system_reset(cancel_async_tasks=True)
        return {"status": "ok", "message": "Sistema resetado e tarefas limpas"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

app.include_router(chat_router)
app.include_router(completion_router)
app.include_router(status_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
