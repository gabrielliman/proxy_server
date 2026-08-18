import json
import time
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import aiohttp
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import os
from typing import Literal
import re
import glob

REQUEST_SEND_TIMES = []
REQUEST_SEND_LOCK = asyncio.Lock()


# ============================================================
# Data structures
# ============================================================

@dataclass
class RequestMetrics:
    ttft: float
    latency: float
    itl: List[float]          # Inter-token latencies (seconds)
    output_tokens: int
    input_tokens: int
    start_time: float
    end_time: float


@dataclass
class ProgramMetrics:
    program_id: str
    requests: List[RequestMetrics]
    waiting_time: float = None
    service_time: float = None


# ============================================================
# Utility functions for metrics
# ============================================================

def percentile(arr, p):
    return float(np.percentile(arr, p)) if arr else None


def safe_div(a, b):
    return a / b if b and b > 0 else None


def _tpot_list(reqs: List[RequestMetrics]) -> List[float]:
    tpot_list = []
    for r in reqs:
        if r.output_tokens > 1:
            tpot_list.append((r.latency)/ (r.output_tokens - 1))
    return tpot_list

def max_request_token_count(reqs) -> int:
    return max((r.input_tokens + r.output_tokens) for r in reqs) if reqs else 0


def summarize_program(prog: ProgramMetrics) -> Dict[str, Any]:
    if not prog.requests:
        return {"program_id": prog.program_id, "num_requests": 0}
    prog_ttfts = min(r.end_time for r in prog.requests) - min(r.start_time for r in prog.requests) if prog.requests else None
    latencies = [r.latency for r in prog.requests if r.latency > 0]    #latencia de uma requisicao é o tempo desde que foi enviada para o escalonador ate receber resposta
    output_tokens = [r.output_tokens for r in prog.requests]
    input_tokens = [r.input_tokens for r in prog.requests]
    max_request_tokens = max_request_token_count(prog.requests)
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
        "total_e2el_s": full_time,
        "start_time": start,
        "end_time": end,
        "num_requests": len(prog.requests),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "prog_ttft_s": prog_ttfts if prog_ttfts else None,
        "waiting_time_s": waiting_time,
        "service_time_s": service_time,
        "max_request_tokens": max_request_tokens,
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


def calculate_percentile_thresholds(
    requests: List[RequestMetrics],
    percentiles: List[int] = [50, 75, 90, 95, 99]
) -> Dict[int, float]:
 
    latencies = [r.latency for r in requests if r.latency > 0]
    
    if not latencies:
        return {p: 0.0 for p in percentiles}
    
    thresholds = {}
    for p in percentiles:
        threshold = percentile(latencies, p)
        thresholds[p] = threshold if threshold is not None else 0.0
    
    return thresholds


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


def separate_all_programs_by_latency(
    program_results: List[ProgramMetrics],
    percentiles: List[int] = [50, 75, 90, 95, 99],
    path: str = "",
    is_baseline: bool = False
) -> Dict[str, Any]:

    results = {}
    for program in program_results:
        results[program.program_id] = separate_requests_by_latency(program, percentiles,path,is_baseline)
    
    return results


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
# ============================================================
# HTTP request logic (returns metrics AND output text)
# ============================================================


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    program_id: str,
    *,
    mode: Literal["chat", "completion"],
    model_name: str,
    tokenizer,
    max_tokens: int,
    temperature: float,
    prompt: str | None = None,
    messages: List[Dict[str, str]] | None = None,
    ) -> Tuple[RequestMetrics, str]:

    if mode == "completion":
        url = f"{base_url}/{program_id}/v1/completions"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        input_text = prompt or ""

    elif mode == "chat":
        url = f"{base_url}/{program_id}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        input_text = "\n".join(m["content"] for m in (messages or []))

    else:
        raise ValueError(f"Invalid mode: {mode}")

    # ---------------------------
    # Token estimation (aligned with your LB)
    # ---------------------------
    input_tokens = len(
        tokenizer(input_text, add_special_tokens=False).input_ids
    )

    start = time.perf_counter()

    try:
        async with session.post(url, json=payload) as resp:
            resp_text = await resp.text()

            if resp.status != 200:
                raise RuntimeError(
                    f"HTTP {resp.status} for program {program_id}: {resp_text}"
                )

            result = json.loads(resp_text)
            end=time.perf_counter()
            latency = end - start

            # ---------------------------
            # Output parsing
            # ---------------------------
            if mode == "completion":
                text = result["choices"][0].get("text", "")
                if not text and "message" in result["choices"][0]:
                    text = result["choices"][0]["message"]["content"]
            else:
                text = result["choices"][0]["message"]["content"]

            output_tokens = len(
                tokenizer(text, add_special_tokens=False).input_ids
            )

            ttft=latency
            itl= [latency/output_tokens-1] if output_tokens>1 else [latency]

            return (
                RequestMetrics(
                    ttft=ttft,
                    latency=latency,
                    itl=itl,
                    output_tokens=output_tokens,
                    input_tokens=input_tokens,
                    start_time=start,
                    end_time=end
                ),
                text,
            )

    except Exception as e:
        print(f"[ERROR] Request failed for program {program_id}: {e}")
        return RequestMetrics(0.0, 0.0, [], 0, 0, 0.0, 0.0), ""



# ============================================================
# Stateful request wrapper
# ============================================================

# ============================================================
# Stateful request wrapper with sliding context window trimming
# ============================================================

async def send_stateful_request(
    session,
    base_url,
    program_id,
    new_user_message,
    *,
    mode: Literal["chat", "completion"],
    model_name,
    tokenizer,
    conversation_state: dict,
    max_tokens=2048,
    temperature=0.0,
    ):
    # Setup your model's maximum limit (matching your 40k max-model-len in vLLM)
    CONTEXT_WINDOW_LIMIT = 40000
    budget = CONTEXT_WINDOW_LIMIT - max_tokens
    # ---------------------------
    # CHAT MODE
    # ---------------------------
    history = conversation_state.get(program_id, [])

    messages = history + [
        {"role": "user", "content": new_user_message}
    ]

    # Dynamically pop oldest messages until the formatted chat structure fits the budget
    while len(messages) > 1:
        try:
            # apply_chat_template accurately counts formatting tags (<|im_start|>, etc.)
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
            num_tokens = len(token_ids)
        except Exception:
            # Fallback text estimation if template processing fails
            combined_text = "\n".join(m["content"] for m in messages)
            num_tokens = len(tokenizer(combined_text, add_special_tokens=False).input_ids)

        if num_tokens <= budget:
            break
        
        # Pop the oldest message (turn) out of the history context
        messages.pop(0)

    rm, text = await send_request(
        session=session,
        base_url=base_url,
        program_id=program_id,
        mode=mode,
        model_name=model_name,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    messages.append({"role": "assistant", "content": text})
    conversation_state[program_id] = messages
    return rm


# ============================================================
# Benchmark orchestration
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
        # print(f"[RateLimiter] Sampled interval: {interval:.4f}s (burstiness={self.burstiness})")
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


async def run_programs(
    program_requests: List[Tuple[str, List[str]]],
    base_url: str,
    model_name: str,
    tokenizer,
    max_concurrency: int,
    rps: float,
    burstiness: float,
    mode: Literal["chat", "completion"],
) -> List[ProgramMetrics]:

    # ----------------------------
    # Concurrency & rate limiting
    # ----------------------------
    if max_concurrency is None or max_concurrency <= 0:
        semaphore = asyncio.Semaphore(1_000_000_000)
    else:
        semaphore = asyncio.Semaphore(max_concurrency)

    rate_limiter = RateLimiter(rps, burstiness)

    total_requests = sum(len(prompts) for _, prompts in program_requests)

    async with aiohttp.ClientSession() as session:

        async def run_single_program(pid, prompts, pbar):
            """
            Executes a single program (chat) SEQUENTIALLY.
            Preserves full chat semantics.
            """
            req_metrics: List[RequestMetrics] = []
            conversation_state = {}

            await rate_limiter.acquire()

            for new_message in prompts:

                # Arrival timestamp for validation (Mantido aqui para marcar a hora exata do envio)
                async with REQUEST_SEND_LOCK:
                    REQUEST_SEND_TIMES.append(time.perf_counter())

                # ----------------------------
                # Execution (SEQUENTIAL per program)
                # ----------------------------
                async with semaphore:
                    rm = await send_stateful_request(
                        session=session,
                        base_url=base_url,
                        program_id=pid,
                        new_user_message=new_message,
                        mode=mode,
                        model_name=model_name,
                        tokenizer=tokenizer,
                        conversation_state=conversation_state,
                    )

                req_metrics.append(rm)
                pbar.update(1)
                async with session.get(f"{base_url}/prog_time/{pid}") as response:
                    # The response is a JSON list containing the two values
                    data = await response.json()
                    if data[0] is not None:
                        waiting_time = float(data[0])
                    else:
                        waiting_time=-1
                    if data[1] is not None:
                        service_time = float(data[1])
                    else:
                        service_time=-1
            return ProgramMetrics(program_id=pid, requests=req_metrics, waiting_time=waiting_time, service_time=service_time)

        # ----------------------------
        # Launch programs concurrently
        # ----------------------------
        program_tasks = []

        with tqdm(total=total_requests, desc="Completed requests") as pbar:
            for pid, prompts in program_requests:
                program_tasks.append(
                    asyncio.create_task(run_single_program(pid, prompts, pbar))
                )

            program_results = await asyncio.gather(*program_tasks)

        return program_results





# ============================================================
# Main benchmark wrapper
# ============================================================

async def benchmark_sharegpt(
    dataset_path: str,
    base_url: str,
    model_name: str,
    limit: int | None = None,
    max_concurrency: int | None = 32,
    rps: float = float("inf"),
    mode: Literal["chat", "completion"] = "completion",
    burstiness: float = 1.0,
    chat_len: int = 0,
    output_json_path: str = "",
    is_baseline: bool = False
):
    print(f"Is baseline run: {is_baseline}")

    try:
        # Attempt to load the primary model's tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=10000)
        
    except Exception as e:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", model_max_length=10000)
    with open(dataset_path, "r", encoding="utf8") as f:
        data = json.load(f)


    # print("\n============================================")
    # print(f"Loaded {len(data)} programs from ShareGPT dataset")
    # print("============================================\n")
    print(len(data))
    program_requests = []
    for conv in data:
        if(len(program_requests)>=limit):
            break
        pid = conv.get("id", "")
        messages = conv.get("conversations", [])
        user_msgs = [m["value"] for m in messages if m.get("from") == "human"]
        if user_msgs:
            # print(pid)
            # if(len(user_msgs)==2):
            if(chat_len>0):
                if(len(user_msgs)>=chat_len):
                    user_msgs=user_msgs[:chat_len]
                    program_requests.append((pid, user_msgs))
            else:
                program_requests.append((pid, user_msgs))

    num_programs = len(program_requests)
    total_requests = sum(len(msgs) for _, msgs in program_requests)

    # print(f"Found {num_programs} programs.")
    # print(f"Total requests: {total_requests}\n")

    start_wall = time.perf_counter()

    program_results = await run_programs(
        program_requests,
        base_url,
        model_name,
        tokenizer,
        max_concurrency,
        rps,
        burstiness,
        mode,
    )

    total_wall = time.perf_counter() - start_wall
    # print(f"\nBenchmark wall-clock duration: {total_wall:.2f}s\n")

    per_program_metrics = [summarize_program(p) for p in program_results]
    
    latency_separation = separate_all_programs_by_latency(
        program_results,
        percentiles=[50, 75, 90, 95, 99],
        is_baseline=is_baseline,
        path=output_json_path
    )
    
    all_requests = [r for p in program_results for r in p.requests]
    full_prog = ProgramMetrics("FULL_DATASET", all_requests)
    full_metrics = summarize_full(full_prog)
    
    full_latency_separation = separate_requests_by_latency(
        full_prog,
        percentiles=[50, 75, 90, 95, 99],
        path=output_json_path,
        is_baseline=is_baseline
    )
    
    # ---- per-program E2EL aggregation ----
    e2el_values = np.array([p["total_e2el_s"] for p in per_program_metrics if p.get("total_e2el_s") is not None], dtype=float)
    if e2el_values.size:
        full_metrics["mean_e2el_per_program"] = float(np.mean(e2el_values))
        full_metrics["median_e2el_per_program"] = float(np.median(e2el_values))
        full_metrics["p75_e2el_per_program"] = float(np.percentile(e2el_values, 75))
        full_metrics["p90_e2el_per_program"] = float(np.percentile(e2el_values, 90))
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
    full_metrics["num_programs"] = num_programs
    full_metrics["total_requests"] = total_requests


    def analyze_request_rate(send_times):
        if len(send_times) < 2:
            return {}

        send_times = sorted(send_times)
        intervals = np.diff(send_times)

        observed_rps = 1.0 / np.mean(intervals)

        return {
            "observed_mean_rps": observed_rps,
            "mean_interval_s": float(np.mean(intervals)),
            "p99_interval_s": float(np.percentile(intervals, 99)),
            "num_samples": len(intervals),
        }


    rate_stats = analyze_request_rate(REQUEST_SEND_TIMES)
    full_metrics["rate_limiter_validation"] = rate_stats

    return per_program_metrics, full_metrics, latency_separation, full_latency_separation



def get_baseline_metrics(output_json_path):
    """
    Recebe o caminho de saída, ex:
        output_plas_kv_token_time_least-total-load_rate4_run1.json
        output_fcfs_least-waiting_rate10_run2.json

    Encontra automaticamente o respectivo baseline:
        baseline_output_plas_kv_token_time_round-robin_rate4_run*.json
        baseline_output_fcfs_round-robin_rate10_run*.json
    """
    filename = os.path.basename(output_json_path)

    # --------------------------------------------------
    # 1) Extrair o sched_suffix correto
    # --------------------------------------------------
    sched_suffix = None
    
    # Lista de prefixos conhecidos baseada no seu script Bash
    known_suffixes = ["fcfs", "plas_service_cumulative", "plas_kv_token_time", "atlas_service_cumulative", "atlas_kv_token_time"]
    
    for suffix in known_suffixes:
        if filename.startswith(f"output_{suffix}_"):
            sched_suffix = suffix
            break
            
    # Fallback genérico caso adicione um scheduler simples (sem underline) no futuro
    if not sched_suffix:
        match_generic = re.search(r"output_([^_]+)_.+_rate", filename)
        if match_generic:
            sched_suffix = match_generic.group(1)
        else:
            raise ValueError(f"Não foi possível determinar o scheduler do arquivo: {filename}")

    # --------------------------------------------------
    # 2) Extrair a taxa (rate)
    # --------------------------------------------------
    rate_match = re.search(r"_rate(\d+)_run", filename)
    if not rate_match:
        raise ValueError(f"Não foi possível encontrar a taxa (rate) no arquivo: {filename}")
    
    rate = rate_match.group(1)

    # --------------------------------------------------
    # 3) Procurar o baseline exato correspondente
    # --------------------------------------------------
    base_dir = os.path.dirname(output_json_path)
    
    # Agora o pattern busca o prefixo composto exato (ex: plas_kv_token_time)
    pattern = os.path.join(
        base_dir,
        f"baseline_output_{sched_suffix}_round-robin_rate{rate}_run*.json"
    )

    files = glob.glob(pattern)

    if len(files) == 0:
        raise ValueError(
            f"Nenhum baseline encontrado para sched_suffix={sched_suffix}, rate={rate}.\n"
            f"Padrão buscado: {pattern}"
        )

    # --------------------------------------------------
    # 4) Agregar métricas dos baselines encontrados
    # --------------------------------------------------
    median_e2el_ms = 0
    p75_e2el_ms = 0
    p90_e2el_ms = 0
    p95_e2el_ms = 0
    p99_e2el_ms = 0

    for full_path in files:
        with open(full_path, "r") as file:
            data = json.load(file)

            median_e2el_ms += data["full_metrics"]["median_e2el_ms"]
            p75_e2el_ms += data["full_metrics"]["p75_e2el_ms"]
            p90_e2el_ms += data["full_metrics"]["p90_e2el_ms"]
            p95_e2el_ms += data["full_metrics"]["p95_e2el_ms"]
            p99_e2el_ms += data["full_metrics"]["p99_e2el_ms"]

    count = len(files)

    median_e2el_ms /= count
    p75_e2el_ms /= count
    p90_e2el_ms /= count
    p95_e2el_ms /= count
    p99_e2el_ms /= count

    return {
        50: median_e2el_ms / 1000,
        75: p75_e2el_ms / 1000,
        90: p90_e2el_ms / 1000,
        95: p95_e2el_ms / 1000,
        99: p99_e2el_ms / 1000,
    }


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=-1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--chat_len", type=int, default=0)
    parser.add_argument(
    "--mode",
    choices=["chat", "completion"],
    default="chat",
    help="Which OpenAI-style API to use",
    )
    parser.add_argument("--is_baseline_run",default=0)
    args = parser.parse_args()


    print(args.is_baseline_run)
    if(args.is_baseline_run=="0"):
        is_baseline=False
    else:
        is_baseline=True

    per_program, full_metrics, latency_separation, full_latency_separation = asyncio.run(
        benchmark_sharegpt(
            dataset_path=args.dataset,
            base_url=args.base_url,
            model_name=args.model,
            mode=args.mode,
            limit=args.limit,
            max_concurrency=args.max_concurrency,
            rps=args.request_rate,
            burstiness=args.burstiness,
            chat_len=args.chat_len,
            output_json_path=args.output_json,
            is_baseline=is_baseline
        )
    )

    print("======== FULL DATASET METRICS ========")
    for k, v in full_metrics.items():
        print(f"{k}: {v}")

    print("\n======== LATENCY PERCENTILE SEPARATION (FULL DATASET) ========")
    for p in sorted(full_latency_separation["metrics_per_percentile"].keys()):
        metrics = full_latency_separation["metrics_per_percentile"][p]
        print(f"\nPercentile P{p}:")
        print(f"  Threshold: {metrics['threshold_ms']:.2f}ms")
        print(f"  Valid Requests: {metrics['num_valid_requests']} ({metrics['valid_request_percentage']:.1f}%)")
        print(f"  Request Goodput: {metrics['request_goodput_rps']:.2f} rps" if metrics['request_goodput_rps'] else "  Request Goodput: N/A")
        print(f"  Token Goodput: {metrics['output_token_goodput']:.2f} tps" if metrics['output_token_goodput'] else "  Token Goodput: N/A")
        print(f"  E2EL - Median: {metrics['median_e2el_ms']:.2f}ms, P99: {metrics['p99_e2el_ms']:.2f}ms" if metrics['median_e2el_ms'] else "  E2EL: N/A")

    print("\n======== TOP 10 PROGRAMS BY NUM_REQUESTS ========")
    top_programs = sorted(
        per_program,
        key=lambda x: x.get("num_requests", 0),
        reverse=True,
    )[:10]

    for entry in top_programs:
        pid = entry["program_id"]
        nreq = entry["num_requests"]
        tin = entry["total_input_tokens"]
        tout = entry["total_output_tokens"]
        print(f"program_id={pid}  num_requests={nreq}  input_tokens={tin}  output_tokens={tout}")

    if args.output_json:
        out_path = args.output_json
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        
        latency_sep_output = {}
        for prog_id, sep_data in latency_separation.items():
            latency_sep_output[prog_id] = {
                "thresholds": sep_data["thresholds"],
                "metrics_per_percentile": sep_data["metrics_per_percentile"],
                "num_percentiles": len(sep_data["metrics_per_percentile"])
            }
        
        full_latency_sep_output = {
            "thresholds": full_latency_separation["thresholds"],
            "metrics_per_percentile": full_latency_separation["metrics_per_percentile"],
        }
        
        to_save = {
            "full_metrics": full_metrics,
            "per_program": per_program,
            "latency_separation": latency_sep_output,
            "full_latency_separation": full_latency_sep_output
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
        print(f"\nSaved metrics JSON to: {out_path}")
