import requests
import time
import os
import concurrent.futures
import statistics
import math

# Configurações iniciais
API_BASE = os.getenv("CUSTOM_API_BASE", "http://localhost:8081")
MODEL = os.getenv("CUSTOM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
PROGRAM_ID = "benchmark_simultaneo"
URL = f"{API_BASE}/{PROGRAM_ID}/v1/chat/completions"
RESET_CACHE_URL = "http://localhost:8005/reset_prefix_cache"

# Aumentado para 30 repetições para validade estatística padrão (Normal / Z-score)
NUM_REPETICOES = 1

def ler_prompt_arquivo(caminho="prompt.txt"):
    """Lê o conteúdo do arquivo txt. Se não existir, cria um de exemplo."""
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            texto = f.read().strip()
            if not texto:
                raise ValueError("O arquivo está vazio.")
            return texto
    except FileNotFoundError:
        print(f"[AVISO] Arquivo '{caminho}' não encontrado.")
        texto_exemplo = "Escreva uma frase filosófica curta sobre programação."
        print(f"Criando '{caminho}' com um texto de exemplo...")
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(texto_exemplo)
        return texto_exemplo

# Lê o arquivo uma única vez na inicialização
TEXTO_DO_PROMPT = ler_prompt_arquivo("prompt.txt")
MENSAGEM = [{"role": "user", "content": TEXTO_DO_PROMPT}]

def resetar_cache():
    """Limpa o prefix cache de forma silenciosa para o loop."""
    try:
        response = requests.post(RESET_CACHE_URL)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        pass 

def fazer_requisicao(n, temperature=1):
    """Retorna a latência e os tokens DE OUTPUT de uma requisição."""
    payload = {
        "model": MODEL,
        "messages": MENSAGEM,
        "max_tokens": 2048,
        "temperature": temperature,
        "n": n
    }
    
    start_time = time.time()
    response = requests.post(URL, json=payload)
    response.raise_for_status()
    end_time = time.time()
    
    result = response.json()
    print(f"Resposta recebida para n={n}: {result.get('choices', [{}])[0].get('message', {}).get('content', '')}...")  # Log parcial da resposta
    # ALTERAÇÃO AQUI: Pegando apenas os tokens gerados (output)
    output_tokens = result.get("usage", {}).get("completion_tokens", 0)
    
    return end_time - start_time, output_tokens

def calcular_estatisticas(dados):
    """Calcula a média e o intervalo de confiança (95%) para n >= 30."""
    if not dados:
        return 0, 0
    if len(dados) == 1:
        return dados[0], 0
        
    media = statistics.mean(dados)
    desvio = statistics.stdev(dados)
    
    # Valor Z para 95% de confiança (distribuição normal)
    z_valor = 1.96 
    
    # Erro padrão da média multiplicado pelo Z-score
    margem_erro = z_valor * (desvio / math.sqrt(len(dados)))
    
    return media, margem_erro

def main():    
    resultados_c1_latencia, resultados_c1_tokens = [], []
    resultados_c2_latencia, resultados_c2_tokens = [], []
    resultados_c3_latencia, resultados_c3_tokens = [], []
    
    for i in range(NUM_REPETICOES):
        print(f"--- Executando Rodada {i + 1}/{NUM_REPETICOES} ---")
        
        try:
            # CASO 1: n=5
            resetar_cache()
            lat, tok = fazer_requisicao(n=5)
            resultados_c1_latencia.append(lat)
            resultados_c1_tokens.append(tok)
            time.sleep(1)
            
            # CASO 2: n=1
            resetar_cache()
            lat, tok = fazer_requisicao(n=1)
            resultados_c2_latencia.append(lat)
            resultados_c2_tokens.append(tok)
            time.sleep(1)
            
            # CASO 3: 5 simultâneas (n=1)
            resetar_cache()
            start_c3 = time.time()
            tokens_c3 = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(fazer_requisicao, 1) for _ in range(5)]
                for future in concurrent.futures.as_completed(futures):
                    _, t = future.result()
                    tokens_c3 += t
            end_c3 = time.time()
            
            resultados_c3_latencia.append(end_c3 - start_c3)
            resultados_c3_tokens.append(tokens_c3)
            time.sleep(1)
            
        except Exception as e:
            print(f"[ERRO] Falha na rodada {i+1}: {e}")
            return

    # --- RELATÓRIO ESTATÍSTICO ---
    print("\n" + "="*60)
    print("RELATÓRIO ESTATÍSTICO FINAL (95% de Confiança - Z=1.96)")
    print("="*60)

    m_lat_1, err_lat_1 = calcular_estatisticas(resultados_c1_latencia)
    m_tok_1 = statistics.mean(resultados_c1_tokens)
    print("\nCASO 1: 1 Requisição com n=5")
    print(f"Latência Média: {m_lat_1:.3f}s (± {err_lat_1:.3f}s)")
    print(f"Tokens de Output Médios : {m_tok_1:.1f}")

    m_lat_2, err_lat_2 = calcular_estatisticas(resultados_c2_latencia)
    m_tok_2 = statistics.mean(resultados_c2_tokens)
    print("\nCASO 2: 1 Requisição com n=1 (Linha de Base)")
    print(f"Latência Média: {m_lat_2:.3f}s (± {err_lat_2:.3f}s)")
    print(f"Tokens de Output Médios : {m_tok_2:.1f}")

    m_lat_3, err_lat_3 = calcular_estatisticas(resultados_c3_latencia)
    m_tok_3 = statistics.mean(resultados_c3_tokens)
    print("\nCASO 3: 5 Requisições simultâneas (n=1)")
    print(f"Latência Média Global: {m_lat_3:.3f}s (± {err_lat_3:.3f}s)")
    print(f"Tokens de Output Médios (Somados) : {m_tok_3:.1f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()