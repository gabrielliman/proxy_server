import os
import argparse
import asyncio
import aiohttp
import time
import numpy as np
import json
import glob
import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import logging

from hotpotqa import HotPotQATask
import models # Para acessar models.PROGRAM_METRICS_REGISTRY e models.REQUEST_SEND_TIMES
from models import gpt_usage, ProgramMetrics
from lats import lats_search
from tot import dfs_search
from rap import mcts_search

# ============================================================
# Utilitários de Métricas (Portados do ShareGPT benchmark)
# ============================================================
def percentile(arr, p):
    return float(np.percentile(arr, p)) if arr else None

def safe_div(a, b):
    return a / b if b and b > 0 else None

def _tpot_list(reqs) -> List[float]:
    tpot_list = []
    for r in reqs:
        if r.output_tokens > 1:
            tpot_list.append((r.latency)/ (r.output_tokens - 1))
    return tpot_list

def summarize_program(prog: ProgramMetrics) -> Dict[str, Any]:
    if not prog.requests:
        return {"program_id": prog.program_id, "num_requests": 0}
    prog_ttfts = min(r.end_time for r in prog.requests) - min(r.start_time for r in prog.requests) if prog.requests else None
    latencies = [r.latency for r in prog.requests if r.latency > 0]    #latencia de uma requisicao é o tempo desde que foi enviada para o escalonador ate receber resposta
    output_tokens = [r.output_tokens for r in prog.requests]
    input_tokens = [r.input_tokens for r in prog.requests]
    start = min(r.start_time for r in prog.requests)
    end = max(r.end_time for r in prog.requests)
    full_time = float(end - start)
    total_latency = sum(latencies) #nao conta o tempo entre a resposta de uma requisicao e um envio de outra (provavelmente mt pequeno pq a fila ta sempre cheia)
    total_output_tokens = sum(output_tokens)
    total_input_tokens = sum(input_tokens)
    total_tokens = total_input_tokens + total_output_tokens
    waiting_time = prog.waiting_time
    service_time = prog.service_time

    tpots = _tpot_list(prog.requests)

    return {
        "program_id": prog.program_id,
        "start_time": start,
        "end_time": end,
        "total_e2el_s": full_time,
        "num_requests": len(prog.requests),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "prog_ttft_s": prog_ttfts if prog_ttfts else None,
        "waiting_time_s": waiting_time,
        "service_time_s": service_time,

        "request_throughput_rps": safe_div(len(latencies), total_latency),
        "output_token_throughput": safe_div(total_output_tokens, total_latency),
        "total_token_throughput": safe_div(total_tokens, total_latency),

        # "mean_ttft_ms": float(np.mean(ttfts)) * 1000 if ttfts else None,
        # "median_ttft_ms": float(np.median(ttfts)) * 1000 if ttfts else None,
        # "p99_ttft_ms": percentile(ttfts, 99) * 1000 if ttfts else None,

        "mean_tpot_ms": float(np.mean(tpots)) * 1000 if tpots else None,
        "median_tpot_ms": float(np.median(tpots)) * 1000 if tpots else None,
        "p99_tpot_ms": percentile(tpots, 99) * 1000 if tpots else None,

        # "mean_itl_ms": float(np.mean(itls)) * 1000 if itls else None,
        # "median_itl_ms": float(np.median(itls)) * 1000 if itls else None,
        # "p99_itl_ms": percentile(itls, 99) * 1000 if itls else None,

        "mean_e2el_ms": float(np.mean(latencies)) * 1000 if latencies else None,
        "median_e2el_ms": float(np.median(latencies)) * 1000 if latencies else None,
        "p75_e2el_ms": percentile(latencies, 75) * 1000 if latencies else None,
        "p90_e2el_ms": percentile(latencies, 90) * 1000 if latencies else None,
        "p95_e2el_ms": percentile(latencies, 95) * 1000 if latencies else None,
        "p99_e2el_ms": percentile(latencies, 99) * 1000 if latencies else None,
    }

def summarize_full(prog: ProgramMetrics) -> Dict[str, Any]:
    ttft = min(r.end_time for r in prog.requests) - min(r.start_time for r in prog.requests) if prog.requests else None
    latencies = [r.latency for r in prog.requests if r.latency > 0]   #latencia de uma requisicao é o tempo desde que foi enviada para o escalonador ate receber resposta
    itls = [t for r in prog.requests for t in r.itl]
    output_tokens = [r.output_tokens for r in prog.requests]
    input_tokens = [r.input_tokens for r in prog.requests]
    total_latency = sum(latencies) #nao conta o tempo entre a resposta de uma requisicao e um envio de outra (provavelmente mt pequeno pq a fila ta sempre cheia)
    total_output_tokens = sum(output_tokens)
    total_input_tokens = sum(input_tokens)
    total_tokens = total_input_tokens + total_output_tokens
    
    tpots = _tpot_list(prog.requests)

    return {
        "program_id": prog.program_id,
        "num_requests": len(prog.requests),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "ttft_s": ttft if ttft else None,

        "request_throughput_rps": safe_div(len(latencies), total_latency),
        "output_token_throughput": safe_div(total_output_tokens, total_latency),
        "total_token_throughput": safe_div(total_tokens, total_latency),

        # "mean_ttft_ms": float(np.mean(ttfts)) * 1000 if ttfts else None,
        # "median_ttft_ms": float(np.median(ttfts)) * 1000 if ttfts else None,
        # "p90_ttft_ms": percentile(ttfts, 90) * 1000 if ttfts else None,
        # "p99_ttft_ms": percentile(ttfts, 99) * 1000 if ttfts else None,

        "mean_tpot_ms": float(np.mean(tpots)) * 1000 if tpots else None,
        "median_tpot_ms": float(np.median(tpots)) * 1000 if tpots else None,
        "p90_tpot_ms": percentile(tpots, 90) * 1000 if tpots else None,
        "p99_tpot_ms": percentile(tpots, 99) * 1000 if tpots else None,

        # "mean_itl_ms": float(np.mean(itls)) * 1000 if itls else None,
        # "median_itl_ms": float(np.median(itls)) * 1000 if itls else None,
        # "p90_itl_ms": percentile(itls, 90) * 1000 if itls else None,
        # "p99_itl_ms": percentile(itls, 99) * 1000 if itls else None,

        "mean_e2el_ms": float(np.mean(latencies)) * 1000 if latencies else None,
        "median_e2el_ms": float(np.median(latencies)) * 1000 if latencies else None,
        "p75_e2el_ms": percentile(latencies, 75) * 1000 if latencies else None,
        "p90_e2el_ms": percentile(latencies, 90) * 1000 if latencies else None,
        "p95_e2el_ms": percentile(latencies, 95) * 1000 if latencies else None,
        "p99_e2el_ms": percentile(latencies, 99) * 1000 if latencies else None,

    }

def get_baseline_metrics(output_json_path):
    if not output_json_path:
        return {50: 0, 75: 0, 90: 0, 95: 0, 99: 0}
        
    filename = os.path.basename(output_json_path)
    sched_suffix = None
    known_suffixes = ["fcfs", "plas_service_cumulative", "plas_kv_token_time", "atlas_service_cumulative", "atlas_kv_token_time"]
    
    for suffix in known_suffixes:
        if filename.startswith(f"output_{suffix}_"):
            sched_suffix = suffix
            break
            
    if not sched_suffix:
        match_generic = re.search(r"output_([^_]+)_.+_rate", filename)
        if match_generic:
            sched_suffix = match_generic.group(1)
        else:
            return {50: 0, 75: 0, 90: 0, 95: 0, 99: 0} # Fallback se não encontrar

    rate_match = re.search(r"_rate(\d+)_run", filename)
    if not rate_match:
        return {50: 0, 75: 0, 90: 0, 95: 0, 99: 0}
    rate = rate_match.group(1)

    base_dir = os.path.dirname(output_json_path)
    pattern = os.path.join(base_dir, f"baseline_output_{sched_suffix}_round-robin_rate{rate}_run*.json")
    files = glob.glob(pattern)

    if len(files) == 0:
        return {50: 0, 75: 0, 90: 0, 95: 0, 99: 0}

    median, p75, p90, p95, p99 = 0, 0, 0, 0, 0
    for full_path in files:
        with open(full_path, "r") as file:
            data = json.load(file)
            median += data["full_metrics"]["median_e2el_ms"]
            p75 += data["full_metrics"]["p75_e2el_ms"]
            p90 += data["full_metrics"]["p90_e2el_ms"]
            p95 += data["full_metrics"]["p95_e2el_ms"]
            p99 += data["full_metrics"]["p99_e2el_ms"]

    count = len(files)
    return {
        50: (median / count) / 1000,
        75: (p75 / count) / 1000,
        90: (p90 / count) / 1000,
        95: (p95 / count) / 1000,
        99: (p99 / count) / 1000,
    }

def calculate_percentile_thresholds(requests, percentiles=[50, 75, 90, 95, 99]):
    latencies = [r.latency for r in requests if r.latency > 0]
    if not latencies: return {p: 0.0 for p in percentiles}
    return {p: (percentile(latencies, p) or 0.0) for p in percentiles}

def separate_requests_by_latency(
    program: ProgramMetrics,
    percentiles: List[int] = [50, 75, 90, 95, 99],
    path: str = "",
    is_baseline: bool = False
) -> Dict[str, Any]:
    
    if not program.requests:
        return {
            "thresholds": {},
            "percentile_buckets": {},
            "metrics_per_percentile": {}
        }
    
    if(is_baseline):
        thresholds = calculate_percentile_thresholds(program.requests, percentiles)
    else:
        thresholds = get_baseline_metrics(path)
    
    percentile_buckets = {}
    for p in sorted(percentiles):
        threshold = thresholds[p]
        valid_requests = [r for r in program.requests if r.latency <= threshold]
        percentile_buckets[p] = valid_requests
    
    metrics_per_percentile = {}
    for p in sorted(percentiles):
        threshold = thresholds[p]
        valid_requests = percentile_buckets[p]
        
        if valid_requests:
            valid_latencies = [r.latency for r in valid_requests if r.latency > 0]
            # valid_ttfts = [r.ttft for r in valid_requests if r.ttft > 0]
            valid_output_tokens = [r.output_tokens for r in valid_requests]
            valid_input_tokens = [r.input_tokens for r in valid_requests]
            # valid_itls = [t for r in valid_requests for t in r.itl]
            valid_tpots = _tpot_list(valid_requests)
            
            total_latency = sum(valid_latencies)
            total_output_tokens = sum(valid_output_tokens)
            total_input_tokens = sum(valid_input_tokens)
            total_tokens = total_input_tokens + total_output_tokens
            
            metrics_per_percentile[p] = {
                "p": p,
                "threshold_ms": threshold * 1000,
                "num_valid_requests": len(valid_requests),
                "valid_request_percentage": (len(valid_requests) / len(program.requests) * 100) if program.requests else 0,
                
                "request_goodput_rps": safe_div(len(valid_requests), total_latency),
                "output_token_goodput": safe_div(total_output_tokens, total_latency),
                "total_token_goodput": safe_div(total_tokens, total_latency),
                
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                
                "mean_e2el_ms": float(np.mean(valid_latencies)) * 1000 if valid_latencies else None,
                "median_e2el_ms": float(np.median(valid_latencies)) * 1000 if valid_latencies else None,
                "p95_e2el_ms": percentile(valid_latencies, 95) * 1000 if valid_latencies else None,
                "p99_e2el_ms": percentile(valid_latencies, 99) * 1000 if valid_latencies else None,
                
                # "mean_ttft_ms": float(np.mean(valid_ttfts)) * 1000 if valid_ttfts else None,
                # "median_ttft_ms": float(np.median(valid_ttfts)) * 1000 if valid_ttfts else None,
                # "p99_ttft_ms": percentile(valid_ttfts, 99) * 1000 if valid_ttfts else None,
                
                "mean_tpot_ms": float(np.mean(valid_tpots)) * 1000 if valid_tpots else None,
                "median_tpot_ms": float(np.median(valid_tpots)) * 1000 if valid_tpots else None,
                "p99_tpot_ms": percentile(valid_tpots, 99) * 1000 if valid_tpots else None,
                
                # "mean_itl_ms": float(np.mean(valid_itls)) * 1000 if valid_itls else None,
                # "median_itl_ms": float(np.median(valid_itls)) * 1000 if valid_itls else None,
                # "p99_itl_ms": percentile(valid_itls, 99) * 1000 if valid_itls else None,
            }
        else:
            metrics_per_percentile[p] = {
                "p": p,
                "threshold_ms": threshold * 1000,
                "num_valid_requests": 0,
                "valid_request_percentage": 0,
                "request_goodput_rps": None,
                "output_token_goodput": None,
                "total_token_goodput": None,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }
    
    return {
        "program_id": program.program_id,
        "thresholds": thresholds,
        "percentile_buckets": percentile_buckets,
        "metrics_per_percentile": metrics_per_percentile
    }

def separate_all_programs_by_latency(program_results, percentiles=[50, 75, 90, 95, 99], path="", is_baseline=False):
    return {program.program_id: separate_requests_by_latency(program, percentiles, path, is_baseline) for program in program_results}

# ============================================================
# Rate Limiter original do LATS
# ============================================================
class RateLimiter:
    def __init__(self, rate: float, burstiness: float = 1.0):
        self.rate = rate
        self.burstiness = burstiness
        self._lock = asyncio.Lock()
        self._mean_interval = 1.0 / rate if rate > 0 and rate != float("inf") else 0.0
        self._last = time.perf_counter()

    def _sample_interval(self) -> float:
        if self.rate <= 0 or self.rate == float("inf"): return 0.0
        if self.burstiness == float("inf"): return self._mean_interval
        theta = 1.0 / (self.rate * self.burstiness)
        return np.random.gamma(shape=self.burstiness, scale=theta)

    async def acquire(self):
        if self.rate <= 0 or self.rate == float("inf"): return
        async with self._lock:
            interval = self._sample_interval()
            now = time.perf_counter()
            target_time = self._last + interval
            wait = target_time - now
            if wait > 0: await asyncio.sleep(wait)
            self._last = target_time

# ============================================================
# Worker Síncrono (Executa em Thread separada)
# ============================================================
def execute_tree_sync(args, task, program_id, idx):
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
# Orquestrador Assíncrono Principal
# ============================================================
async def run_async_orchestrator(args):
    task = HotPotQATask()
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

    rate_limiter = RateLimiter(args.program_rate, args.burstiness)
    loop = asyncio.get_running_loop()
    
    tasks = []
    task_accs = []
    
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=1000000) as executor:
        for i in range(args.task_start_index, args.task_end_index):
            await rate_limiter.acquire()
            program_id = f"prog_{i}"
            future = loop.run_in_executor(executor, execute_tree_sync, args, task, program_id, i)
            tasks.append(future)

        results = await asyncio.gather(*tasks)

    total_wall = time.perf_counter() - start_time

    for prog_id, idx, em, reward in results:
        em_val = 0 if em is None else em
        task_accs.append(em_val)
        
    cnt_avg = sum(task_accs) / len(task_accs) if task_accs else 0
    print(f"\nFinalizado! Total processado: {len(results)}")
    print(f"Média de Exact Match (EM): {cnt_avg}")
    print(f"Tempo total (Orquestrador): {total_wall:.2f}s")
    print('usage_so_far', gpt_usage(args.backend))

    # ============================================================
    # COMPILAÇÃO DAS MÉTRICAS (ShareGPT Style)
    # ============================================================
    print("\nCompilando métricas de requisição...")
    base_url = os.getenv("CUSTOM_API_BASE", "http://localhost:8000")
    program_results = []
    for pid, reqs in models.PROGRAM_METRICS_REGISTRY.items():
        async with aiohttp.ClientSession() as session:    
            async with session.get(f"{base_url}/prog_time/{pid}") as response:
                data = await response.json()
                if data[0] is not None:
                    waiting_time = float(data[0])
                else:
                    waiting_time=-1
                if data[1] is not None:
                    service_time = float(data[1])
                else:
                    waiting_time=-1
        program_results.append(ProgramMetrics(program_id=pid, requests=reqs, waiting_time=waiting_time, service_time=service_time))

    per_program_metrics = [summarize_program(p) for p in program_results]
    
    latency_separation = separate_all_programs_by_latency(
        program_results, percentiles=[50, 75, 90, 95, 99],
        is_baseline=args.is_baseline_run, path=args.output_json
    )
    
    all_requests = [r for p in program_results for r in p.requests]
    full_prog = ProgramMetrics("FULL_DATASET", all_requests)
    full_metrics = summarize_full(full_prog)
    
    full_latency_separation = separate_requests_by_latency(
        full_prog, percentiles=[50, 75, 90, 95, 99],
        path=args.output_json, is_baseline=args.is_baseline_run
    )

    # Aggregação E2EL por programa
    e2el_values = np.array([p["total_e2el_s"] for p in per_program_metrics if p.get("total_e2el_s") is not None], dtype=float)
    if e2el_values.size:
        full_metrics["mean_e2el_per_program"] = float(np.mean(e2el_values))
        full_metrics["median_e2el_per_program"] = float(np.median(e2el_values))
        full_metrics["p95_e2el_per_program"] = float(np.percentile(e2el_values, 95))
        full_metrics["p99_e2el_per_program"] = float(np.percentile(e2el_values, 99))


    ttft_values = np.array(
        [p["prog_ttft_s"] for p in per_program_metrics if p.get("prog_ttft_s") is not None],
        dtype=float
    )
    if ttft_values.size:
        full_metrics["mean_ttft_per_program"] = float(np.mean(ttft_values))
        full_metrics["median_ttft_per_program"] = float(np.median(ttft_values))
        full_metrics["p95_ttft_per_program"] = float(np.percentile(ttft_values, 95))
        full_metrics["p99_ttft_per_program"] = float(np.percentile(ttft_values, 99))
    
        waiting_time = np.array(
        [p["waiting_time_s"] for p in per_program_metrics if p.get("waiting_time_s") is not None],
        dtype=float
    )
        
    if waiting_time.size:
        full_metrics["mean_waiting_time_per_program"] = float(np.mean(waiting_time))
        full_metrics["median_waiting_time_per_program"] = float(np.median(waiting_time))
        full_metrics["p95_waiting_time_per_program"] = float(np.percentile(waiting_time, 95))
        full_metrics["p99_waiting_time_per_program"] = float(np.percentile(waiting_time, 99))
        full_metrics["total_waiting_time_s"] = float(np.sum(waiting_time))

    service_time = np.array(
        [p["service_time_s"] for p in per_program_metrics if p.get("service_time_s") is not None],
        dtype=float
    )
    if service_time.size:
        full_metrics["mean_service_time_per_program"] = float(np.mean(service_time))
        full_metrics["median_service_time_per_program"] = float(np.median(service_time))
        full_metrics["p95_service_time_per_program"] = float(np.percentile(service_time, 95))
        full_metrics["p99_service_time_per_program"] = float(np.percentile(service_time, 99))
        full_metrics["total_service_time_s"] = float(np.sum(service_time))


    full_metrics["total_e2el_s"] = total_wall
    full_metrics["num_programs"] = len(program_results)
    full_metrics["total_requests"] = len(all_requests)

    # Validação do Rate Limiter
    def analyze_request_rate(send_times):
        if len(send_times) < 2: return {}
        send_times = sorted(send_times)
        intervals = np.diff(send_times)
        return {
            "observed_mean_rps": 1.0 / np.mean(intervals),
            "mean_interval_s": float(np.mean(intervals)),
            "p99_interval_s": float(np.percentile(intervals, 99)),
            "num_samples": len(intervals),
        }

    full_metrics["rate_limiter_validation"] = analyze_request_rate(models.REQUEST_SEND_TIMES)

    # Prints Finais
    print("\n======== FULL DATASET METRICS ========")
    for k, v in full_metrics.items():
        print(f"{k}: {v}")

    print("\n======== LATENCY PERCENTILE SEPARATION (FULL DATASET) ========")
    for p in sorted(full_latency_separation["metrics_per_percentile"].keys()):
        metrics = full_latency_separation["metrics_per_percentile"][p]
        print(f"\nPercentile P{p}:")
        print(f"  Threshold: {metrics['threshold_ms']:.2f}ms")
        print(f"  Valid Requests: {metrics['num_valid_requests']} ({metrics.get('valid_request_percentage',0):.1f}%)")
        print(f"  Request Goodput: {metrics.get('request_goodput_rps') or 'N/A'}")
        print(f"  Token Goodput: {metrics.get('output_token_goodput') or 'N/A'}")
        print(f"  E2EL - Median: {metrics.get('median_e2el_ms') or 'N/A'}, P99: {metrics.get('p99_e2el_ms') or 'N/A'}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True) if os.path.dirname(args.output_json) else None
        
        latency_sep_output = {
            prog_id: {
                "thresholds": sep_data["thresholds"],
                "metrics_per_percentile": sep_data["metrics_per_percentile"],
                "num_percentiles": len(sep_data["metrics_per_percentile"])
            } for prog_id, sep_data in latency_separation.items()
        }
        
        to_save = {
            "full_metrics": full_metrics,
            "per_program": per_program_metrics,
            "latency_separation": latency_sep_output,
            "full_latency_separation": {
                "thresholds": full_latency_separation["thresholds"],
                "metrics_per_percentile": full_latency_separation["metrics_per_percentile"],
            }
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
        print(f"\nSaved metrics JSON to: {args.output_json}")

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
    args.add_argument('--program-rate', type=float, default=float("inf"), help='Taxa de início de novos programas por segundo')
    args.add_argument('--burstiness', type=float, default=1.0, help='Controle de rajada (1.0 = Poisson)')
    args.add_argument('--weight', action='store_true', help='Toggle weights_only security feature for PyTorch 2.6+')
    
    # NOVOS ARGUMENTOS DE MÉTRICA
    args.add_argument('--output-json', type=str, default=None, help='Caminho para salvar o JSON gerado')
    args.add_argument('--is_baseline_run', type=int, default=0, help='0 = Falso, 1 = Verdadeiro (para cálculo de Thresholds)')

    parsed = args.parse_args()
    parsed.is_baseline_run = bool(parsed.is_baseline_run)
    return parsed

if __name__ == '__main__':
    args = parse_args()
    print(args)
    asyncio.run(run_async_orchestrator(args))