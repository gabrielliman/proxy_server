from fastapi import APIRouter
from routing.selector import backend_metrics, metrics_lock
from request_queue.manager import get_queue_state
import asyncio
from routing.process_table import PROCESS_TABLE

router = APIRouter()


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