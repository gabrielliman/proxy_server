from pathlib import Path
import re
import json
import glob
import os
import statistics
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

def limpar_logs(pasta_alvo):
    caminho = Path(pasta_alvo)
    
    # Verifica se o caminho realmente é uma pasta válida
    if not caminho.is_dir():
        print(f"A pasta '{pasta_alvo}' não foi encontrada.")
        return

    # Percorre todos os arquivos .txt dentro da pasta
    for arquivo in caminho.glob("*.txt"):
        try:
            # 1. Apaga se estiver vazio (0 bytes)
            if arquivo.stat().st_size == 0:
                print(f"Apagado (Vazio): {arquivo.name}")
                arquivo.unlink()
                continue
            
            # 2. Lê as linhas do arquivo
            with open(arquivo, 'r', encoding='utf-8') as f:
                linhas = f.readlines()
            
            # Verifica se tem exatamente 8 linhas
            if len(linhas) == 8:
                # A 7ª linha fica no índice 6. Usamos strip() para remover o \n do final
                linha_7 = linhas[6].strip()
                
                # Normaliza os espaços da linha lida (converte espaços duplos/invisíveis em espaços simples)
                # para garantir que a comparação não falhe por causa de formatação boba
                linha_7_normalizada = re.sub(r'\s+', ' ', linha_7)
                texto_alvo_normalizado = "INFO: Application shutdown complete."
                
                if linha_7_normalizada == texto_alvo_normalizado:
                    print(f"Apagado (Shutdown log): {arquivo.name}")
                    arquivo.unlink()
                    
        except Exception as e:
            print(f"Erro ao ler o arquivo {arquivo.name}: {e}")

def analisar_preferred_engines(caminho_json):
    """
    Lê um arquivo JSON e calcula métricas sobre a chave 'preferred_engines'.
    """
    try:
        # Carrega os dados do arquivo JSON
        with open(caminho_json, 'r', encoding='utf-8') as arquivo:
            dados = json.load(arquivo)
            
        # Dicionário para armazenar a contagem por programa
        contagem_por_programa = {}
        
        for programa, metricas in dados.items():
            # Usa .get() para evitar erro caso a chave não exista em algum programa
            engines = metricas.get("preferred_engines", [])
            contagem_por_programa[programa] = len(engines)
            
        valores = list(contagem_por_programa.values())
        
        if not valores:
            print("Nenhum dado válido encontrado no JSON.")
            return None
            
        # Calcula a média
        media = statistics.mean(valores)
        
        # O cálculo do desvio padrão amostral (stdev) exige pelo menos 2 elementos.
        # Se você quiser o desvio padrão populacional, basta usar statistics.pstdev
        if len(valores) > 1:
            desvio_padrao = statistics.stdev(valores)
        else:
            desvio_padrao = 0.0
            
        # Exibição dos resultados
        print("=== Métricas de Preferred Engines ===")
        print(f"Total de programas: {len(valores)}")
        print(f"Média por programa: {media:.2f}")
        print(f"Desvio padrão:      {desvio_padrao:.2f}\n")
        
        return
    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_json}' não foi encontrado.")
    except json.JSONDecodeError:
        print("Erro: O arquivo não é um JSON válido.")

def plot_metrics(csv_file, title=None):

    # Carregar os dados do CSV
    df = pd.read_csv(csv_file)
    
    # Converter timestamp para datetime (opcional, mas ajuda na formatação do eixo X)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Ordenar por tempo para garantir que a linha do tempo esteja correta no gráfico
    df = df.sort_values(by='datetime')

    # --- INÍCIO DA FILTRAGEM ---
    # Agrupa por backend e verifica o valor máximo das colunas alvo
    backend_stats = df.groupby('backend')[['kv_cache_percent', 'running', 'waiting']].max()
    
    # Mantém apenas os backends onde pelo menos uma dessas colunas é diferente de 0
    active_backends = backend_stats[(backend_stats != 0).any(axis=1)].index
    
    # Filtra o dataframe original para manter apenas esses backends ativos
    df = df[df['backend'].isin(active_backends)]
    # --- FIM DA FILTRAGEM ---

    # Obter a lista de backends únicos (agora sem os que são totalmente zero)
    backends = df['backend'].unique()

    # Criar a figura com 5 subplots (5 linhas, 1 coluna)
    # Aumentado o figsize de (32, 16) para (32, 20) para acomodar o novo plot
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(32, 20), sharex=True)
    
    ax_kv, ax_running, ax_waiting, ax_global, ax_global_running = axes

    # Plotar as métricas por backend
    for backend in backends:
        backend_df = df[df['backend'] == backend]
        
        window_size = 30 # Tente valores maiores, como 20 ou 50, se continuar muito poluído

        # Plotando com média móvel
        ax_kv.plot(backend_df['datetime'], backend_df['kv_cache_percent'].rolling(window_size).mean(), label=backend, linewidth=1.5)
        ax_running.plot(backend_df['datetime'], backend_df['running'].rolling(window_size).mean(), label=backend, linewidth=1.5)
        ax_waiting.plot(backend_df['datetime'], backend_df['waiting'].rolling(window_size).mean(), label=backend, linewidth=1.5)
        
    # Configurações do gráfico de KV Cache Percent
    ax_kv.set_title('KV Cache Percent por Backend')
    ax_kv.set_ylabel('KV Cache (%)')
    ax_kv.grid(True)
    
    # Configurações do gráfico de Running
    ax_running.set_title('Running Requests por Backend')
    ax_running.set_ylabel('Running')
    ax_running.grid(True)
    
    # Configurações do gráfico de Waiting
    ax_waiting.set_title('Waiting Requests por Backend')
    ax_waiting.set_ylabel('Waiting')
    ax_waiting.grid(True)

    # Para as métricas globais (incomplete_count e proxy_waiting_count)
    global_df = df.groupby('datetime')[['incomplete_count', 'proxy_waiting_count']].max().reset_index()
    
    # --- NOVO CÁLCULO: Running requests globais ---
    global_df['global_running'] = global_df['incomplete_count'] - global_df['proxy_waiting_count']
    
    # Plotando métricas globais originais
    ax_global.plot(global_df['datetime'], global_df['incomplete_count'], label='Incomplete Count', color='red', linewidth=2)
    ax_global.plot(global_df['datetime'], global_df['proxy_waiting_count'], label='Proxy Waiting Count', color='purple', linewidth=2)
    
    # Configurações do gráfico global original
    ax_global.set_title('Métricas Globais (Incomplete & Proxy Waiting)')
    ax_global.set_ylabel('Count')
    ax_global.grid(True)
    ax_global.legend()

    # Plotando a nova métrica global de Running Requests
    ax_global_running.plot(global_df['datetime'], global_df['global_running'], label='Running Requests (Incomplete - Waiting)', color='green', linewidth=2)
    
    # Configurações do novo gráfico
    ax_global_running.set_title('Global Running Requests')
    ax_global_running.set_xlabel('Tempo')
    ax_global_running.set_ylabel('Running')
    ax_global_running.grid(True)
    ax_global_running.legend()

    # Colocar a legenda dos backends fora do gráfico para não poluir a visão
    if len(backends) > 0:
        handles, labels = ax_kv.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), title="Backends")

    # --- ADICIONA O TÍTULO GERAL (OPCIONAL) ---
    if title:
        # y=0.98 coloca o título um pouco mais para cima para não colidir com o gráfico
        fig.suptitle(title, fontsize=24, fontweight='bold', y=0.98)

    # Ajustar o layout para que os rótulos e legendas não fiquem cortados
    # Se houver título, deixa um pequeno respiro no topo (0.96) em vez de preencher tudo (1)
    plt.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])
    
    # Mostrar o gráfico (Corrigido para incluir os parênteses)
    plt.show()

def gerar_tabela_resumo_latex(resultados, y_metric="total_e2el_s"):
    """Gera uma tabela unificada apenas com os Valores Principais de todos os cenários."""
    # Identifica todas as colunas (combinações de run_type e load_balancer)
    colunas = []
    for run_type in ["Baseline", "Standard"]:
        if run_type in resultados:
            for lb in sorted(resultados[run_type].keys()):
                colunas.append((run_type, lb))
                
    if not colunas:
        return ""

    num_colunas = 1 + len(colunas)
    alinhamento = "l" + "r" * (num_colunas - 1)
    
    latex = []
    latex.append(r"\begin{table}[htpb]")
    latex.append(r"\centering")
    metric_tex = y_metric.replace('_', r'\_')
    latex.append(f"\\caption{{Resumo Geral do Benchmark ({metric_tex} em segundos)}}")
    latex.append(f"\\begin{{tabular}}{{{alinhamento}}}")
    
    latex.append(r"\hline")
    
    # Monta o cabeçalho APENAS com o Load Balancer (removido Baseline/Standard)
    header = r"\textbf{Run}"
    for run_type, lb in colunas:
        nome_coluna = f"{lb}".replace('_', r'\_')
        header += f" & \\textbf{{{nome_coluna}}}"
    header += r" \\"
    latex.append(header)
    latex.append(r"\hline")
    
    valores_por_coluna = {col: [] for col in colunas}

    todas_runs = [run for rt in resultados.values() for lb in rt.values() for run in lb.keys()]
    max_run = max(todas_runs) if todas_runs else 1
    
    # Preenche as linhas (runs 1 a 10)
    for run in range(1, max_run + 1):
        linha = f"{run:02d}"
        for col in colunas:
            run_type, lb = col
            val = resultados[run_type][lb].get(run, {}).get("metric_value")
            
            if val is not None:
                linha += f" & {val:.4f}"
                valores_por_coluna[col].append(val)
            else:
                linha += " & - "
        linha += r" \\"
        latex.append(linha)
        
    latex.append(r"\hline")
    
    linha_media = "Média"
    for col in colunas:
        vals = valores_por_coluna[col]
        if vals:
            linha_media += f" & {statistics.mean(vals):.4f}"
        else:
            linha_media += " & - "
    linha_media += r" \\"
    latex.append(linha_media)
    
    linha_std = "Desvio Padrão"
    for col in colunas:
        vals = valores_por_coluna[col]
        if len(vals) > 1:
            linha_std += f" & {statistics.stdev(vals):.4f}"
        else:
            linha_std += " & - "
    linha_std += r" \\"
    latex.append(linha_std)
    
    latex.append(r"\hline")
    latex.append(r"\end{tabular}%")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)

def results(folder, y_metric):
    resultados = defaultdict(lambda: defaultdict(dict))

    padrao_baseline = os.path.join(folder, "baseline_output_atlas_service_cumulative_*_rate*_run*.json")
    padrao_normal = os.path.join(folder, "output_atlas_service_cumulative_*_rate*_run*.json")
    
    arquivos = glob.glob(padrao_baseline) + glob.glob(padrao_normal)

    if not arquivos:
        print(f"Nenhum arquivo encontrado na pasta: {folder}")
        return

    print(f"Encontrados {len(arquivos)} arquivos. Processando...\n")

    for caminho_arquivo in arquivos:
        nome_arquivo = os.path.basename(caminho_arquivo)
        
        run_type = "Baseline" if nome_arquivo.startswith("baseline_output") else "Standard"

        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            
            full_metrics = dados.get("full_metrics", {})
            metric_value = full_metrics.get(y_metric)
            load_balancer = full_metrics.get("load_balancer", "desconhecido")
            
            programs_data = {}
            for prog in dados.get("per_program", []):
                prog_id = prog.get("program_id")
                prog_metric = prog.get(y_metric)
                if prog_id and prog_metric is not None:
                    programs_data[prog_id] = prog_metric

            match = re.search(r'run(\d+)\.json', nome_arquivo)
            if match:
                run_num = int(match.group(1))
                resultados[run_type][load_balancer][run_num] = {
                    "metric_value": metric_value,
                    "programs": programs_data
                }

        except Exception as e:
            print(f"Erro ao processar o arquivo {nome_arquivo}: {e}")

    tabela_resumo = gerar_tabela_resumo_latex(resultados, y_metric)

    print(tabela_resumo)

def gerar_tabela_hit_rate_latex(resultados):
    """Gera uma tabela com os valores de Hit Rate do Prefix Cache por run e o resumo (Média +- Desvio)."""
    colunas = []
    for run_type in ["Baseline", "Standard"]:
        if run_type in resultados:
            for lb in sorted(resultados[run_type].keys()):
                colunas.append((run_type, lb))
                
    if not colunas:
        return ""

    num_colunas = 1 + len(colunas)
    alinhamento = "l" + "c" * (num_colunas - 1)
    
    latex = []
    latex.append(r"\begin{table}[htpb]")
    latex.append(r"\centering")
    latex.append(r"\caption{Resumo do Hit Rate Global do Prefix Cache}")
    latex.append(f"\\begin{{tabular}}{{{alinhamento}}}")
    latex.append(r"\hline")
    
    # Cabeçalho com o nome do Load Balancer
    header = r"\textbf{Run}"
    for run_type, lb in colunas:
        nome_coluna = f"{lb}".replace('_', r'\_')
        header += f" & \\textbf{{{nome_coluna}}}"
    header += r" \\"
    latex.append(header)
    latex.append(r"\hline")
    
    # Identifica dinamicamente o número máximo de runs encontradas
    max_run = 0
    for rt in resultados:
        for lb in resultados[rt]:
            if resultados[rt][lb]:
                max_run = max(max_run, max(resultados[rt][lb].keys()))
    
    # Preenche as linhas das runs
    for run in range(1, max_run + 1):
        linha = f"{run:02d}"
        for col in colunas:
            run_type, lb = col
            val = resultados[run_type][lb].get(run, {}).get("hit_rate")
            
            if val is not None:
                linha += f" & {val:.4f}"
            else:
                linha += " & - "
        linha += r" \\"
        latex.append(linha)
        
    latex.append(r"\hline")
    
    # Linha de resumo: Média \pm Desvio Padrão
    linha_resumo = r"\textbf{Média $\pm$ DP}"
    for col in colunas:
        run_type, lb = col
        vals = [d["hit_rate"] for d in resultados[run_type][lb].values()]
        
        if vals:
            media = statistics.mean(vals)
            if len(vals) > 1:
                desvio = statistics.stdev(vals)
                linha_resumo += f" & ${media:.4f} \\pm {desvio:.4f}$"
            else:
                linha_resumo += f" & ${media:.4f}$"
        else:
            linha_resumo += " & - "
            
    linha_resumo += r" \\"
    latex.append(linha_resumo)
    
    latex.append(r"\hline")
    latex.append(r"\end{tabular}%")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)

def results_hit_rate(folder):
    resultados = defaultdict(lambda: defaultdict(dict))

    padrao_baseline = os.path.join(folder, "baseline_output_atlas_service_cumulative_*_rate*_run*.json")
    padrao_normal = os.path.join(folder, "output_atlas_service_cumulative_*_rate*_run*.json")
    
    arquivos = glob.glob(padrao_baseline) + glob.glob(padrao_normal)

    if not arquivos:
        print(f"Nenhum arquivo encontrado na pasta: {folder}")
        return

    print(f"Encontrados {len(arquivos)} arquivos. Processando Hit Rate...\n")

    for caminho_arquivo in arquivos:
        nome_arquivo = os.path.basename(caminho_arquivo)
        run_type = "Baseline" if nome_arquivo.startswith("baseline_output") else "Standard"

        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            
            full_metrics = dados.get("full_metrics", {})
            load_balancer = full_metrics.get("load_balancer", "desconhecido")
            
            # --- Extração e Cálculo do Prefix Cache Hit Rate ---
            prefix_cache = full_metrics.get("prefix_cache", {})
            total_queries = 0
            total_hits = 0
            
            # Itera sobre todas as engines para pegar o acumulado da run
            for engine, stats in prefix_cache.items():
                total_queries += stats.get("queries", 0)
                total_hits += stats.get("hits", 0)
            
            # Calcula o hit rate global. Se não houver queries, assume 0.
            run_hit_rate = (total_hits / total_queries) if total_queries > 0 else 0.0

            match = re.search(r'run(\d+)\.json', nome_arquivo)
            if match:
                run_num = int(match.group(1))
                resultados[run_type][load_balancer][run_num] = {
                    "hit_rate": run_hit_rate
                }

        except Exception as e:
            print(f"Erro ao processar o arquivo {nome_arquivo}: {e}")

    tabela_resumo = gerar_tabela_hit_rate_latex(resultados)

    print(tabela_resumo)
