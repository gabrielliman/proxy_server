import os
import json
import glob
import numpy as np

def calcular_metricas_wiki(pasta_origem, pasta_destino=None):
    """
    Lê todos os JSONs de 'pasta_origem', calcula as métricas do wiki 
    baseado nos dados de 'per_program' e atualiza 'full_metrics'.
    Se 'pasta_destino' não for fornecida, sobrescreve os arquivos originais.
    """
    
    # Cria a pasta de destino se ela for especificada e não existir
    if pasta_destino and not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
        
    caminho_busca = os.path.join(pasta_origem, '*.json')
    arquivos_json = glob.glob(caminho_busca)
    
    if not arquivos_json:
        print(f"Nenhum arquivo JSON encontrado na pasta: {pasta_origem}")
        return

    for arquivo in arquivos_json:
        with open(arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            
        tempos_busca = []
        num_buscas = []
        
        # 1. Coletar os dados de cada programa
        for prog in dados.get("per_program", []):
            if "wiki_search_time_s" in prog:
                tempos_busca.append(prog["wiki_search_time_s"])
            if "wiki_num_searches" in prog:
                num_buscas.append(prog["wiki_num_searches"])
                
        # 2. Calcular e injetar as métricas se houver dados
        if tempos_busca and num_buscas:
            full_metrics = dados.get("full_metrics", {})
            
            # Métricas para wiki_search_time_s
            # Convertendo para float() ou int() padrão do Python para evitar erros de serialização do numpy no json.dump
            full_metrics["total_wiki_search_time_s"] = float(np.sum(tempos_busca))
            full_metrics["mean_wiki_search_time_s"] = float(np.mean(tempos_busca))
            full_metrics["median_wiki_search_time_s"] = float(np.median(tempos_busca))
            full_metrics["p90_wiki_search_time_s"] = float(np.percentile(tempos_busca, 90))
            full_metrics["p95_wiki_search_time_s"] = float(np.percentile(tempos_busca, 95))
            full_metrics["p99_wiki_search_time_s"] = float(np.percentile(tempos_busca, 99))
            
            # Métricas para wiki_num_searches
            full_metrics["total_wiki_num_searches"] = int(np.sum(num_buscas))
            full_metrics["mean_wiki_num_searches"] = float(np.mean(num_buscas))
            full_metrics["median_wiki_num_searches"] = float(np.median(num_buscas))
            full_metrics["p90_wiki_num_searches"] = float(np.percentile(num_buscas, 90))
            full_metrics["p95_wiki_num_searches"] = float(np.percentile(num_buscas, 95))
            full_metrics["p99_wiki_num_searches"] = float(np.percentile(num_buscas, 99))
            
            # Atualiza o objeto full_metrics no dicionário principal
            dados["full_metrics"] = full_metrics
            
        # 3. Salvar o JSON atualizado
        caminho_salvamento = os.path.join(pasta_destino, os.path.basename(arquivo)) if pasta_destino else arquivo
        
        with open(caminho_salvamento, 'w', encoding='utf-8') as f:
            json.dump(dados, f, indent=2, ensure_ascii=False)
            
        print(f"Processado e atualizado: {os.path.basename(arquivo)}")

# Exemplo de uso:
# Substitua 'caminho/para/seus/jsons' pelo diretório real.
# Se quiser evitar sobrescrever, preencha o segundo argumento.
pasta_dos_arquivos = "Testes8Engines/200prog"
pasta_saida_modificada = "Testes8Engines/200prog" # Deixe None se quiser sobrescrever

calcular_metricas_wiki(pasta_dos_arquivos, pasta_saida_modificada)