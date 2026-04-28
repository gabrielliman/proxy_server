import json
import asyncio
import aiohttp
import argparse
import time
from typing import List, Tuple
from dataclasses import dataclass
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

@dataclass
class RequestMetrics:
    prompt_text: str  # NOVO: Guardará o texto completo do contexto
    kv_token_time: float

@dataclass
class ProgramMetrics:
    program_id: str
    requests: List[RequestMetrics]

async def send_chat_request(
    session: aiohttp.ClientSession,
    base_url: str,
    program_id: str,
    messages: List[dict],
    raw_prompt_text: str, # Recebe o texto formatado para salvar
    model_name: str,
    tokenizer,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> Tuple[RequestMetrics, str]:
    """
    Send a chat request maintaining context and compute KV Token-Time.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions" # Corrigido para o endpoint padrão do vLLM

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        async with session.post(url, json=payload) as resp:
            resp_text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(
                    f"HTTP {resp.status} for program {program_id}: {resp_text}"
                )

            result = json.loads(resp_text)

            generated_text = result["choices"][0]["message"].get("content", "")

            usage = result.get("usage", {})
            p = usage.get("prompt_tokens", 0)
            d = usage.get("completion_tokens", 0)

            if p == 0 or d == 0:
                d = len(tokenizer(generated_text, add_special_tokens=False).input_ids)
                prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                p = len(tokenizer(prompt_str, add_special_tokens=False).input_ids)

            kv_token_time = (p * d) + ((d ** 2) / 2)

            return RequestMetrics(prompt_text=raw_prompt_text, kv_token_time=kv_token_time), generated_text

    except Exception as e:
        print(f"[ERROR] Request failed for program {program_id}: {e}")
        return RequestMetrics(prompt_text=raw_prompt_text, kv_token_time=0.0), ""


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

        program_metrics: List[ProgramMetrics] = []

        async def run_single_program(
            pid: str,
            prompts: List[str],
            start_delay: float,
            pbar: tqdm,
        ) -> ProgramMetrics:
            if start_delay > 0:
                await asyncio.sleep(start_delay)

            req_metrics: List[RequestMetrics] = []
            chat_history = [] 
            
            for user_prompt in prompts:
                chat_history.append({"role": "user", "content": user_prompt})
                
                # NOVO: Concatena o contexto atual de forma simples para usar no seu TF-IDF depois
                raw_prompt_text = "\n".join([m["content"] for m in chat_history])
                
                async with semaphore:
                    rm, assistant_reply = await send_chat_request(
                        session=session,
                        base_url=base_url,
                        program_id=pid,
                        messages=chat_history,
                        raw_prompt_text=raw_prompt_text, # Passa o texto para salvar
                        model_name=model_name,
                        tokenizer=tokenizer,
                    )
                
                req_metrics.append(rm)
                
                if assistant_reply:
                    chat_history.append({"role": "assistant", "content": assistant_reply})
                    
                pbar.update(1)

            return ProgramMetrics(program_id=pid, requests=req_metrics)

        tasks = []
        with tqdm(total=total_requests, desc="Completed requests") as pbar:
            for (pid, prompts), delay in zip(program_requests, delays):
                tasks.append(
                    asyncio.create_task(
                        run_single_program(pid, prompts, delay, pbar)
                    )
                )

            program_metrics = await asyncio.gather(*tasks)

    return program_metrics

async def benchmark_sharegpt(
    dataset_path: str,
    base_url: str,
    model_name: str,
    output_json_path: str,
    limit: int | None = None,
    max_concurrency: int = 32,
    rps: float = float("inf"),
) -> None:
    
    print(f"Carregando tokenizador para o modelo: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(dataset_path, "r", encoding="utf8") as f:
        data = json.load(f)

    if limit is not None:
        data = data[:limit]

    print("\n============================================")
    print(f"Dataset carregado: {len(data)} programas (conversas).")
    print("============================================\n")

    program_requests: List[Tuple[str, List[str]]] = []
    for conv in data:
        pid = conv.get("id", "")
        messages = conv.get("conversations", [])
        user_msgs = [m["value"] for m in messages if m.get("from") == "human"]
        if user_msgs:
            program_requests.append((pid, user_msgs))

    num_programs = len(program_requests)
    total_requests = sum(len(msgs) for _, msgs in program_requests)

    print(f"Encontrados {num_programs} programas válidos.")
    print(f"Total de turnos de conversa (requisições) a enviar: {total_requests}\n")

    start_wall = time.perf_counter()
    
    program_results: List[ProgramMetrics] = await run_programs(
        program_requests=program_requests,
        base_url=base_url,
        model_name=model_name,
        tokenizer=tokenizer,
        max_concurrency=max_concurrency,
        rps=rps,
    )
    
    total_wall = time.perf_counter() - start_wall

    print(f"\nDuração total do Benchmark: {total_wall:.2f} segundos\n")

    # ============================================================
    # Formata e Salva os Resultados (AGORA COM O TEXTO DO PROMPT)
    # ============================================================
    print(f"Salvando métricas e prompts em: {output_json_path}")
    
    output_data = []
    for prog in program_results:
        prog_dict = {
            "program_id": prog.program_id,
            "requests": [
                {
                    "prompt": req.prompt_text, # Texto com o contexto acumulado
                    "kv_token_time": req.kv_token_time
                } 
                for req in prog.requests
            ]
        }
        output_data.append(prog_dict)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("Finalizado com sucesso!")

# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Benchmark para KV Token-Time com Contexto")
    
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", type=str, default="kv_metrics.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--rps", type=float, default=float("inf"))

    args = parser.parse_args()

    asyncio.run(
        benchmark_sharegpt(
            dataset_path=args.dataset,
            base_url=args.base_url,
            model_name=args.model,
            output_json_path=args.output_json,
            limit=args.limit,
            max_concurrency=args.max_concurrency,
            rps=args.rps,
        )
    )



# export HF_HOME="/scratch/global/huggingface_cache/huggingface"
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model "Qwen/Qwen3-4B" --port 8105 --max-num-seqs 256 --dtype bfloat16 --max-model-len 40000 --gpu-memory-utilization 0.9
# python kv_token_time.py --dataset /scratch/global/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --base-url http://localhost:8105 --model Qwen/Qwen3-4B --output-json kv_token_qwen_100token.json --max-concurrency 256



# export HF_HOME="/scratch/global/huggingface_cache/huggingface"
# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model "meta-llama/Llama-3.1-8B-Instruct" --port 8106 --max-num-seqs 256 --dtype bfloat16 --max-model-len 40000 --gpu-memory-utilization 0.9
# python kv_token_time.py --dataset /scratch/global/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --base-url http://localhost:8106 --model meta-llama/Llama-3.1-8B-Instruct --output-json kv_token_llama_100token.json --max-concurrency 256

