import os
import argparse
import asyncio
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

from hotpotqa import HotPotQATask
from models import gpt_usage
from lats import lats_search
from tot import dfs_search
from rap import mcts_search

# ============================================================
# Rate Limiter (Exato modelo do benchmark_stateful.py)
# ============================================================
class RateLimiter:
    """
    Async rate limiter with optional burstiness.
    
    If burstiness == 1 → Poisson (exponential intervals)
    If burstiness < 1 → bursty
    If burstiness > 1 → more uniform
    If burstiness == inf → deterministic interval
    """

    def __init__(self, rate: float, burstiness: float = 1.0):
        self.rate = rate
        self.burstiness = burstiness
        self._lock = asyncio.Lock()

        if rate > 0 and rate != float("inf"):
            self._mean_interval = 1.0 / rate
        else:
            self._mean_interval = 0.0

        self._last = time.perf_counter()

    def _sample_interval(self) -> float:
        """
        Sample next inter-arrival interval.
        """
        if self.rate <= 0 or self.rate == float("inf"):
            return 0.0

        if self.burstiness == float("inf"):
            return self._mean_interval

        # Gamma sampling
        theta = 1.0 / (self.rate * self.burstiness)
        interval=np.random.gamma(
            shape=self.burstiness,
            scale=theta
        )
        return interval

    async def acquire(self):
        if self.rate <= 0 or self.rate == float("inf"):
            return

        async with self._lock:
            interval = self._sample_interval()

            now = time.perf_counter()
            target_time = self._last + interval

            wait = target_time - now
            if wait > 0:
                await asyncio.sleep(wait)

            self._last = target_time

# ============================================================
# Worker Síncrono (Executa em Thread separada)
# ============================================================
def execute_tree_sync(args, task, program_id, idx):
    """
    Esta função roda em uma thread dedicada. 
    Uma vez iniciada, não possui limites de taxa locais e roda
    na velocidade máxima que a API responder.
    """
    logging.info(f"[{program_id}] Iniciando busca para questão {idx}...")
    
    if args.algorithm == 'lats':
        state, value, all_nodes, reward, em = lats_search(args, task, idx, args.iterations, True, program_id=program_id)
    elif args.algorithm == 'tot':
        state, value, all_nodes, reward, em = dfs_search(args, task, idx, args.iterations)
    elif args.algorithm == 'rap':
        state, value, all_nodes, reward, em = mcts_search(args, task, idx, args.iterations)
    else:
        raise Exception("Search algorithm option not valid")
        
    return program_id, idx, em, reward

# ============================================================
# Orquestrador Assíncrono
# ============================================================
async def run_async_orchestrator(args):
    task = HotPotQATask()
    print(task)
    
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

    # Inicializa o rate limiter usando a taxa de PROGRAMAS
    rate_limiter = RateLimiter(args.program_rate, args.burstiness)
    loop = asyncio.get_running_loop()
    
    tasks = []
    task_accs = []
    
    start_time = time.perf_counter()

    # ThreadPoolExecutor com max_workers altíssimo para garantir que não haja
    # gargalo local (sem limite de concorrência artificial)
    with ThreadPoolExecutor(max_workers=1000000) as executor:
        for i in range(args.task_start_index, args.task_end_index):
            
            # O Rate Limiter atua EXCLUSIVAMENTE aqui: na chegada (nascimento) do programa
            await rate_limiter.acquire()
            
            program_id = f"prog_{i}"
            
            # Submete a execução síncrona para uma thread em background
            future = loop.run_in_executor(
                executor, 
                execute_tree_sync, 
                args, task, program_id, i
            )
            tasks.append(future)

        # Aguarda todas as árvores/programas terminarem
        results = await asyncio.gather(*tasks)

    # Processamento de resultados
    for prog_id, idx, em, reward in results:
        em_val = 0 if em is None else em
        task_accs.append(em_val)
        
    cnt_avg = sum(task_accs) / len(task_accs) if task_accs else 0
    print(f"\nFinalizado! Total processado: {len(results)}")
    print(f"Média de Exact Match (EM): {cnt_avg}")
    print(f"Tempo total (Orquestrador): {time.perf_counter() - start_time:.2f}s")
    print('usage_so_far', gpt_usage(args.backend))

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default=None)
    args.add_argument('--temperature', type=float, default=1.0)
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'], default='standard')
    args.add_argument('--n_generate_sample', type=int, default=1)  
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--iterations', type=int, default=50)
    args.add_argument('--log', type=str, default='logs/lats_test.log')
    args.add_argument('--algorithm', type=str, choices=['lats', 'rap', 'tot'], default='lats')
    
    # NOVOS ARGUMENTOS
    args.add_argument('--program-rate', type=float, default=float("inf"), help='Taxa de início de novos programas por segundo')
    args.add_argument('--burstiness', type=float, default=1.0, help='Controle de rajada (1.0 = Poisson)')

    return args.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    asyncio.run(run_async_orchestrator(args))