import os
import requests
import warnings
import concurrent.futures
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List
from transformers import AutoTokenizer

completion_tokens = prompt_tokens = 0
MAX_TOKENS = 4000

# Criamos um Lock para atualizar as variáveis globais de contagem de forma segura entre threads
_token_lock = threading.Lock()

# ==============================================================================
# Estruturas e Globais para Métricas de Benchmark (ShareGPT style)
# ==============================================================================
@dataclass
class RequestMetrics:
    ttft: float
    latency: float
    itl: List[float]
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

# Registry global para armazenar as métricas de cada programa/thread invisivelmente
PROGRAM_METRICS_REGISTRY = defaultdict(list)
_metrics_lock = threading.Lock()

REQUEST_SEND_TIMES = []
_send_time_lock = threading.Lock()

_global_tokenizer = None
_tokenizer_lock = threading.Lock()
def get_tokenizer():
    """
    Carrega o tokenizador exato do modelo uma única vez, de forma thread-safe.
    """
    global _global_tokenizer
    with _tokenizer_lock:
        if _global_tokenizer is None:
            # Pega o modelo definido na variável de ambiente (ou usa um fallback)
            model_name = os.getenv("CUSTOM_MODEL", "gpt2") 
            try:
                # Ignorando o limite rígido para permitir contagem de prompts longos
                _global_tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=100000)
            except Exception as e:
                print(f"[AVISO] Falha ao carregar AutoTokenizer para {model_name}. Usando fallback. Erro: {e}")
                from transformers import GPT2Tokenizer
                _global_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", model_max_length=100000)
                
    return _global_tokenizer

def tokens_in_text(text):
    """
    Conta os tokens de forma precisa usando o tokenizador real do modelo.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

# ==============================================================================
# Configurações da sua API Customizada
# ==============================================================================
API_BASE = os.getenv("CUSTOM_API_BASE", "http://localhost:8000")
DEFAULT_MODEL = os.getenv("CUSTOM_MODEL", "seu-modelo-aqui")

def gpt(prompt, model=None, temperature=1.0, max_tokens=1000, n=1, stop=None, program_id="default_prog") -> list:
    """
    Wrapper para prompts em texto simples. Adicionado o parâmetro program_id.
    """
    system_instruction = (
        "You are a strict data retrieval agent. You MUST solve the question answering task with interleaving Thought, Action, and Observation steps. "
        "You must NEVER write stories, narratives, or conversational text. "
        "Your output must STRICTLY follow this exact format for every single step:\n"
        "Thought: <your reasoning>\n"
        "Action: <Search[entity] OR Lookup[keyword] OR Finish[answer]>"
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt}
    ]
    
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop, program_id=program_id)

def _single_request(url, payload, program_id):
    """
    Função auxiliar para executar uma única requisição HTTP.
    Agora rastreia E2EL, TTFT, TPOT, ITL e salva no registry.
    """
    # Estima os tokens de input baseando-se nas mensagens enviadas
    input_text = "\n".join([m.get("content", "") for m in payload.get("messages", [])])
    input_tokens = tokens_in_text(input_text)
    
    start_time = time.perf_counter()
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        end_time = time.perf_counter()
        result = response.json()
        
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = result.get("usage", {})
        
        # Lógica de cálculo de tempo e output tokens
        latency = end_time - start_time
        output_tokens = tokens_in_text(content) if content else 0
        
        if output_tokens > 0:
            ttft = 0.001 # Aproximação para APIs sem streaming
            if output_tokens > 1:
                per_token = (latency - ttft) / (output_tokens - 1)
                itl = [per_token] * (output_tokens - 1)
            else:
                itl = []
        else:
            ttft = latency
            itl = []
            
        rm = RequestMetrics(
            ttft=ttft,
            latency=latency,
            itl=itl,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            start_time=start_time,
            end_time=end_time
        )
        
        # Salva as métricas silenciosamente
        with _metrics_lock:
            PROGRAM_METRICS_REGISTRY[program_id].append(rm)
            
        return content, usage
        
    except Exception as e:
        print(f"[ERRO API] Falha na requisição para programa {program_id}: {e}")
        end_time = time.perf_counter()
        rm = RequestMetrics(0.0, 0.0, [], 0, input_tokens, start_time, end_time)
        with _metrics_lock:
            PROGRAM_METRICS_REGISTRY[program_id].append(rm)
        return "", {}

def chatgpt(messages, model=None, temperature=1.0, max_tokens=1000, n=1, stop=None, program_id="default_prog") -> list:
    """
    Faz 'n' requisições HTTP síncronas em paralelo para a API customizada usando Threads.
    """
    global completion_tokens, prompt_tokens
    
    target_model = model if model else DEFAULT_MODEL
    url = f"{API_BASE}/{program_id}/v1/chat/completions"
    
    payload = {
        "model": target_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": 1  
    }
    
    if stop:
        payload["stop"] = stop if isinstance(stop, list) else [stop]

    outputs = []
    max_workers = min(n, 50) 
    
    # Registra o tempo de "envio" das requisições para a validação do Rate Limiter Global
    with _send_time_lock:
        for _ in range(n):
            REQUEST_SEND_TIMES.append(time.perf_counter())
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_single_request, url, payload, program_id) for _ in range(n)]
        
        for future in concurrent.futures.as_completed(futures):
            content, usage = future.result()
            outputs.append(content)
            
            if usage:
                with _token_lock:
                    completion_tokens += usage.get("completion_tokens", 0)
                    prompt_tokens += usage.get("prompt_tokens", 0)
            
    return outputs
    
def gpt_usage(backend="custom"):
    global completion_tokens, prompt_tokens
    cost = 0.0 
    return {
        "completion_tokens": completion_tokens, 
        "prompt_tokens": prompt_tokens, 
        "cost": cost
    }