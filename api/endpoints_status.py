from fastapi import APIRouter
from routing.selector import backend_metrics, metrics_lock
from request_queue.manager import get_queue_state
import asyncio
from routing.process_table import PROCESS_TABLE
import os
import signal
from fastapi import HTTPException, Body
from routing.load_balancer import LOAD_BALANCER
from pydantic import BaseModel

router = APIRouter()

class ProgramCompleteRequest(BaseModel):
    program_id: str

@router.get("/status")
async def status():
    with metrics_lock:
        return backend_metrics


@router.get("/queue")
async def queue_state():
    return get_queue_state()


@router.get("/processes")
async def processes():
    # return the current process table snapshot
    print("Fetching process table snapshot")
    return PROCESS_TABLE.list_processes()

@router.get("/processes_summary")
async def processes_summary():
    print("Fetching process table summary")
    return PROCESS_TABLE.sum_processes()

@router.get("/processes_detail")
async def processes_detail():
    from routing.process_table import PROCESS_TABLE
    return PROCESS_TABLE.list_processes()

@router.get("/waiting_time/{program_id}")
async def get_waiting_time(program_id: str):
    # return the waiting time for a specific program
    # print(f"Fetching waiting time for program: {program_id}")
    return PROCESS_TABLE.get_waiting_time(program_id)

@router.get("/service_time/{program_id}")
async def get_service_time(program_id: str):
    # return the service time for a specific program
    # print(f"Fetching service time for program: {program_id}")
    return PROCESS_TABLE.get_service_time(program_id)

@router.get("/prog_time/{program_id}")
async def get_program_time(program_id: str):
    # return the waiting time for a specific program
    # print(f"Fetching program time for program: {program_id}")
    return PROCESS_TABLE.get_waiting_time(program_id), PROCESS_TABLE.get_service_time(program_id)

@router.get("/incomplete_requests")
async def incomplete_requests_count():
    # Retorna o número de requisições na tabela de processo que ainda não concluíram
    # print("Fetching incomplete requests count")
    count = PROCESS_TABLE.get_incomplete_requests_count()
    return {"incomplete_requests": count}

@router.get("/waiting_requests")
async def waiting_requests_count():
    # print("Fetching waiting requests count")
    count = PROCESS_TABLE.get_waiting_requests_count()
    return {"waiting_requests": count}

from routing.queue_manager import get_queue
@router.delete("/purge_waiting")
async def purge_waiting():
    """
    Esvazia a fila de requisições pendentes, limpa a tabela de processos
    e bloqueia o proxy para não receber mais requisições vLLM.
    """
    cleared_queue_items = 0
    try:
        queue = get_queue("global_cluster")
        while not queue.empty():
            queue.get_nowait()
            cleared_queue_items += 1
    except Exception as e:
        print(f"[PURGE ERROR] Falha ao limpar a fila do asyncio: {e}")

    purge_stats = PROCESS_TABLE.purge_waiting_and_stale()

    PROCESS_TABLE.accepting_requests = False
    
    return {
        "status": "success",
        "accepting_new_requests": PROCESS_TABLE.accepting_requests,
        "cleared_queue_items": cleared_queue_items,
        "removed_waiting_threads": purge_stats["removed_threads"],
        "removed_programs_count": len(purge_stats["removed_programs"]),
        "removed_program_ids": purge_stats["removed_programs"]
    }

@router.post("/program/complete")
async def complete_program(payload: ProgramCompleteRequest):
    """Notifica o Load Balancer que o programa terminou para limpar o estado/afinidade."""
    try:
        await LOAD_BALANCER.on_program_complete(payload.program_id)
        return {"status": "success", "message": f"Program {payload.program_id} completed"}
    except Exception as e:
        import traceback
        traceback.print_exc()  # Prints the full stack trace to server logs
        raise HTTPException(status_code=500, detail=str(e))