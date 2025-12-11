import json
import time
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import aiohttp
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm


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


@dataclass
class ProgramMetrics:
    program_id: str
    requests: List[RequestMetrics]


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
            tpot_list.append((r.latency - r.ttft) / (r.output_tokens - 1))
    return tpot_list


def summarize_program(prog: ProgramMetrics) -> Dict[str, Any]:
    ttfts = [r.ttft for r in prog.requests if r.ttft > 0]
    latencies = [r.latency for r in prog.requests if r.latency > 0]
    itls = [t for r in prog.requests for t in r.itl]
    output_tokens = [r.output_tokens for r in prog.requests]
    input_tokens = [r.input_tokens for r in prog.requests]

    total_latency = sum(latencies)
    total_output_tokens = sum(output_tokens)
    total_input_tokens = sum(input_tokens)
    total_tokens = total_input_tokens + total_output_tokens

    tpots = _tpot_list(prog.requests)

    return {
        "program_id": prog.program_id,
        "num_requests": len(prog.requests),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,

        "request_throughput_rps": safe_div(len(latencies), total_latency),
        "output_token_throughput": safe_div(total_output_tokens, total_latency),
        "total_token_throughput": safe_div(total_tokens, total_latency),

        "mean_ttft_ms": float(np.mean(ttfts)) * 1000 if ttfts else None,
        "median_ttft_ms": float(np.median(ttfts)) * 1000 if ttfts else None,
        "p99_ttft_ms": percentile(ttfts, 99) * 1000 if ttfts else None,

        "mean_tpot_ms": float(np.mean(tpots)) * 1000 if tpots else None,
        "median_tpot_ms": float(np.median(tpots)) * 1000 if tpots else None,
        "p99_tpot_ms": percentile(tpots, 99) * 1000 if tpots else None,

        "mean_itl_ms": float(np.mean(itls)) * 1000 if itls else None,
        "median_itl_ms": float(np.median(itls)) * 1000 if itls else None,
        "p99_itl_ms": percentile(itls, 99) * 1000 if itls else None,

        "mean_e2el_ms": float(np.mean(latencies)) * 1000 if latencies else None,
        "median_e2el_ms": float(np.median(latencies)) * 1000 if latencies else None,
        "p99_e2el_ms": percentile(latencies, 99) * 1000 if latencies else None,
    }


# ============================================================
# HTTP request logic (returns metrics AND output text)
# ============================================================

async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    program_id: str,
    prompt: str,
    model_name: str,
    tokenizer,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> Tuple[RequestMetrics, str]:

    url = f"{base_url}/{program_id}/v1/completions"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    input_tokens = len(
        tokenizer(prompt, add_special_tokens=False).input_ids
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
            full_latency = time.perf_counter() - start

            text = ""
            try:
                text = result["choices"][0].get("text", "")
                if not text and "message" in result["choices"][0]:
                    text = result["choices"][0]["message"].get("content", "")
            except:
                raise RuntimeError(f"Unexpected response format: {result}")

            output_tokens = len(
                tokenizer(text, add_special_tokens=False).input_ids
            )

            if output_tokens > 0:
                ttft = 0.001
                if output_tokens > 1:
                    per_token = (full_latency - ttft) / (output_tokens - 1)
                    itl = [per_token] * (output_tokens - 1)
                else:
                    itl = []
            else:
                ttft = full_latency
                itl = []

            return (
                RequestMetrics(
                    ttft=ttft,
                    latency=full_latency,
                    itl=itl,
                    output_tokens=output_tokens,
                    input_tokens=input_tokens,
                ),
                text
            )

    except Exception as e:
        print(f"[ERROR] Request failed for program {program_id}: {e}")
        return RequestMetrics(0.0, 0.0, [], 0, 0), ""


# ============================================================
# Stateful request wrapper
# ============================================================

async def send_stateful_request(
    session,
    base_url,
    program_id,
    new_user_message,
    model_name,
    tokenizer,
    conversation_state: Dict[str, str],
    max_tokens=2048,
    temperature=0.0,
):

    history = conversation_state.get(program_id, "")

    if history == "":
        prompt = new_user_message
    else:
        prompt = history + "\n\n" + new_user_message

    rm, output_text = await send_request(
        session=session,
        base_url=base_url,
        program_id=program_id,
        prompt=prompt,
        model_name=model_name,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    updated_history = prompt #+ "\n\n" + output_text
    conversation_state[program_id] = updated_history

    return rm


# ============================================================
# Benchmark orchestration
# ============================================================

async def run_programs(
    program_requests: List[Tuple[str, List[str]]],
    base_url: str,
    model_name: str,
    tokenizer,
    max_concurrency: int,
    rps: float,
) -> List[ProgramMetrics]:

    if max_concurrency is None or max_concurrency <= 0:
        semaphore = asyncio.Semaphore(1_000_000_000)
    else:
        semaphore = asyncio.Semaphore(max_concurrency)

    total_requests = sum(len(prompts) for _, prompts in program_requests)

    async with aiohttp.ClientSession() as session:

        if rps is None or rps == float("inf") or rps <= 0:
            delays = [0.0 for _ in program_requests]
        else:
            delays = [i / rps for i in range(len(program_requests))]
        async def run_single_program(pid, prompts, start_delay, pbar):

                if start_delay > 0:
                    await asyncio.sleep(start_delay)

                req_metrics = []
                conversation_state = {}

                for new_message in prompts:
                    # === DEBUG PRINT DO QUE ESTÁ SENDO AO MODELO ===
                    print(f"\n📨 [PROGRAM {pid}] New user message:")
                    print("   ", new_message.replace("\n", " ")[:300], "...")

                    history_preview = conversation_state.get(pid, "")
                    print(f"   [History length = {len(history_preview)} chars]")

                    async with semaphore:
                        rm = await send_stateful_request(
                            session=session,
                            base_url=base_url,
                            program_id=pid,
                            new_user_message=new_message,
                            model_name=model_name,
                            tokenizer=tokenizer,
                            conversation_state=conversation_state,
                        )

                    req_metrics.append(rm)
                    pbar.update(1)

                return ProgramMetrics(program_id=pid, requests=req_metrics)


        tasks = []
        with tqdm(total=total_requests, desc="Completed requests") as pbar:
            for (pid, prompts), delay in zip(program_requests, delays):
                tasks.append(asyncio.create_task(
                    run_single_program(pid, prompts, delay, pbar)
                ))

            return await asyncio.gather(*tasks)


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
    burstiness: float = 1.0,
):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(dataset_path, "r", encoding="utf8") as f:
        data = json.load(f)

    if limit is not None:
        data = data[:limit]

    print("\n============================================")
    print(f"Loaded {len(data)} programs from ShareGPT dataset")
    print("============================================\n")

    program_requests = []
    for conv in data:
        pid = conv.get("id", "")
        messages = conv.get("conversations", [])
        user_msgs = [m["value"] for m in messages if m.get("from") == "human"]
        if user_msgs:
            program_requests.append((pid, user_msgs))

    num_programs = len(program_requests)
    total_requests = sum(len(msgs) for _, msgs in program_requests)

    print(f"Found {num_programs} programs.")
    print(f"Total requests: {total_requests}\n")

    start_wall = time.perf_counter()

    program_results = await run_programs(
        program_requests,
        base_url,
        model_name,
        tokenizer,
        max_concurrency,
        rps,
    )

    total_wall = time.perf_counter() - start_wall
    print(f"\nBenchmark wall-clock duration: {total_wall:.2f}s\n")

    per_program_metrics = [summarize_program(p) for p in program_results]

    all_requests = [r for p in program_results for r in p.requests]
    full_prog = ProgramMetrics("FULL_DATASET", all_requests)
    full_metrics = summarize_program(full_prog)

    full_metrics["benchmark_wall_time_s"] = total_wall
    full_metrics["num_programs"] = num_programs
    full_metrics["total_requests"] = total_requests

    return per_program_metrics, full_metrics


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--output-json", type=str, default=None)

    args = parser.parse_args()

    per_program, full_metrics = asyncio.run(
        benchmark_sharegpt(
            dataset_path=args.dataset,
            base_url=args.base_url,
            model_name=args.model,
            limit=args.limit,
            max_concurrency=args.max_concurrency,
            rps=args.request_rate,
            burstiness=args.burstiness,
        )
    )

    print("======== FULL DATASET METRICS ========")
    for k, v in full_metrics.items():
        print(f"{k}: {v}")

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
        to_save = {"full_metrics": full_metrics, "per_program": per_program}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
        print(f"\nSaved metrics JSON to: {out_path}")


# python benchmark_stateful.py \
#   --base-url http://localhost:8080 \
#   --dataset /scratch/global/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
#   --model meta-llama/Llama-3.1-8B-Instruct \
#   --limit 100 \
#   --output-json out.json
# 