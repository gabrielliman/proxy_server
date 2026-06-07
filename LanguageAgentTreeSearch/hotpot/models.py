import os
import requests
import warnings
import concurrent.futures
import threading
from transformers import GPT2Tokenizer

completion_tokens = prompt_tokens = 0
MAX_TOKENS = 4000

# Criamos um Lock para atualizar as variáveis globais de contagem de forma segura entre threads
_token_lock = threading.Lock()

def tokens_in_text(text):
    """
    Accurately count the number of tokens in a string using the GPT-2 tokenizer.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokens = tokenizer.encode(text)
    return len(tokens)

# ==============================================================================
# Configurações da sua API Customizada
# Configure via variáveis de ambiente ou altere diretamente aqui
# ==============================================================================
API_BASE = os.getenv("CUSTOM_API_BASE", "http://localhost:8000")
DEFAULT_MODEL = os.getenv("CUSTOM_MODEL", "seu-modelo-aqui")

def gpt(prompt, model=None, temperature=1.0, max_tokens=1000, n=1, stop=None, program_id="default_prog") -> list:
    """
    Wrapper para prompts em texto simples. Adicionado o parâmetro program_id.
    """
    # Isolate the system instruction from the raw prompt
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
    Retorna uma tupla: (texto_gerado, dict_de_uso_de_tokens)
    """
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Pega a primeira escolha, já que forçamos n=1 no payload desta thread
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = result.get("usage", {})
        return content, usage
        
    except Exception as e:
        print(f"[ERRO API] Falha na requisição para programa {program_id}: {e}")
        try:
            print(f"Detalhes: {response.text}")
        except:
            pass
        return "", {}

def chatgpt(messages, model=None, temperature=1.0, max_tokens=1000, n=1, stop=None, program_id="default_prog") -> list:
    """
    Faz 'n' requisições HTTP síncronas em paralelo para a API customizada usando Threads.
    """
    global completion_tokens, prompt_tokens
    
    target_model = model if model else DEFAULT_MODEL
    url = f"{API_BASE}/{program_id}/v1/chat/completions"
    
    # Prepara o payload base fixando n=1 para CADA requisição
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
    
    # Limita o número de threads simultâneas caso n seja um número muito grande
    max_workers = min(n, 50) 
    
    # Inicia o executor de threads para rodar em paralelo
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Lança as n tarefas de forma simultânea
        futures = [executor.submit(_single_request, url, payload, program_id) for _ in range(n)]
        
        # Coleta os resultados à medida que ficam prontos
        for future in concurrent.futures.as_completed(futures):
            content, usage = future.result()
            outputs.append(content)
            
            # Atualiza o log de uso de tokens com Lock para evitar problemas de concorrência
            if usage:
                with _token_lock:
                    completion_tokens += usage.get("completion_tokens", 0)
                    prompt_tokens += usage.get("prompt_tokens", 0)
            
    return outputs
    
def gpt_usage(backend="custom"):
    """
    Retorna o uso total. Como a API é local, definimos o custo em dólar como 0.
    """
    global completion_tokens, prompt_tokens
    cost = 0.0 
    return {
        "completion_tokens": completion_tokens, 
        "prompt_tokens": prompt_tokens, 
        "cost": cost
    }