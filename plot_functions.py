from pathlib import Path
import re
import json
import glob
import os
import statistics
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_tokens_vs_duration(json_path):
    """
    Reads a JSON file and plots a scatter graph of Total Tokens vs Duration 
    for each program.
    """
    # Load the JSON data
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Extract the program details
    programs = data.get("per_program", [])
    
    total_tokens_list = []
    duration_list = []
    
    for prog in programs:
        # Get token counts and calculate the total
        input_tokens = prog.get("total_input_tokens", 0)
        output_tokens = prog.get("total_output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        
        # Get the total end-to-end latency (duration) in seconds
        duration = prog.get("total_e2el_s", 0)
        
        total_tokens_list.append(total_tokens)
        duration_list.append(duration)
        
    if not total_tokens_list:
        print("No valid program data found in the JSON.")
        return

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    
    # Using alpha=0.7 helps visualize overlapping points if there are any
    plt.scatter(total_tokens_list, duration_list, color='darkorange', alpha=0.7, edgecolors='k')
    
    # Configure labels and title
    plt.title("Relationship Between Total Tokens and Program Duration", fontsize=14, fontweight='bold')
    plt.xlabel("Total Tokens (Input + Output)", fontsize=12)
    plt.ylabel("Program Duration (Seconds)", fontsize=12)
    
    # Add a grid for easier reading
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def process_token_latency_metrics(folder_path):
    """
    Iterates over all JSON files in the specified folder, calculates per-program 
    token latency (total_e2el_s / total_output_tokens), and adds aggregate 
    metrics to 'full_metrics'.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # Read the JSON file
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")
                    continue
            
            programs = data.get("per_program", [])
            latencies = []

            # Calculate token latency for each program and update the item
            for prog in programs:
                # Use total_e2el_s to match your JSON schema exactly
                e2e_s = prog.get("total_e2el_s")
                output_tokens = prog.get("total_output_tokens")

                # Safeguard against missing keys, None values, or division by zero
                if e2e_s is not None and output_tokens and output_tokens > 0:
                    latency = float(e2e_s / output_tokens)
                    prog["program_token_latency"] = latency
                    latencies.append(latency)

            if latencies:
                # Calculate summary metrics across all valid programs
                metrics_to_add = {
                    "mean_program_token_latency": float(np.mean(latencies)),
                    "max_program_token_latency": float(np.max(latencies)),
                    "min_program_token_latency": float(np.min(latencies)),
                    "p99_program_token_latency": float(np.percentile(latencies, 99)),
                    "p95_program_token_latency": float(np.percentile(latencies, 95)),
                    "median_program_token_latency": float(np.median(latencies))
                }
                
                # Ensure full_metrics dictionary exists
                if "full_metrics" not in data:
                    data["full_metrics"] = {}
                    
                # Add the aggregated metrics to full_metrics
                data["full_metrics"].update(metrics_to_add)
                
                # Save the updated JSON back to the file
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=2)
                    
def process_iterations_metrics(folder_path):
    """
    Iterates over all JSON files in the specified folder, calculates metrics for 
    'iterations_used' across all programs, and adds them to 'full_metrics'.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # Read the JSON file
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")
                    continue
            
            # Extract iterations_used from the per_program array
            programs = data.get("per_program", [])
            iterations = [prog.get("iterations_used") for prog in programs if "iterations_used" in prog]
            
            if iterations:
                # Calculate the requested metrics
                metrics_to_add = {
                    "mean_iterations_used": float(np.mean(iterations)),
                    "max_iterations_used": int(np.max(iterations)),
                    "min_iterations_used": int(np.min(iterations)),
                    "p99_iterations_used": float(np.percentile(iterations, 99)),
                    "p95_iterations_used": float(np.percentile(iterations, 95)),
                    "median_iterations_used": float(np.median(iterations))
                }
                
                # Ensure full_metrics exists
                if "full_metrics" not in data:
                    data["full_metrics"] = {}
                    
                # Add the calculated metrics to full_metrics
                data["full_metrics"].update(metrics_to_add)
                
                # Save the updated JSON back to the file
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=2)
                    
            # print(f"Processed: {filename}")

def merge_engine_request_counts(folder_path: str):
    """
    Scans a folder for output JSONs, matches them with their corresponding
    processes_summary JSONs, aggregates engine request counts, and injects
    them into the full_metrics section of the output JSON.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".json") and "output_" in filename:
            output_filepath = os.path.join(folder_path, filename)
            summary_filename = filename.replace("output_", "processes_summary_")
            summary_filepath = os.path.join(folder_path, summary_filename)
            
            if os.path.exists(summary_filepath):
                try:
                    with open(summary_filepath, 'r') as f_summary:
                        summary_data = json.load(f_summary)
                        
                    aggregated_engine_counts = defaultdict(int)
                    
                    for prog_id, prog_metrics in summary_data.items():
                        engine_counts = prog_metrics.get("engine_request_counts", {})
                        
                        for engine_url, count in engine_counts.items():
                            aggregated_engine_counts[engine_url] += count
                    
                    with open(output_filepath, 'r') as f_output:
                        output_data = json.load(f_output)
                        
                    if "full_metrics" not in output_data:
                        output_data["full_metrics"] = {}
                        
                    output_data["full_metrics"]["engine_request_counts"] = dict(aggregated_engine_counts)
                    
                    with open(output_filepath, 'w') as f_output:
                        json.dump(output_data, f_output, indent=2)
                        
                    # print(f"Successfully updated: {filename} using data from {summary_filename}")
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for {filename} or {summary_filename}: {e}")
                except Exception as e:
                    print(f"Unexpected error processing {filename}: {e}")

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
                # print(f"Apagado (Vazio): {arquivo.name}")
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
                    # print(f"Apagado (Shutdown log): {arquivo.name}")
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

# def plot_metrics(csv_file, title=None, json_path=None):
#     # Inferir o caminho do JSON se não for fornecido
#     if json_path is None:
#         json_path = csv_file.replace('_kv_cache.csv', '.json')

#     # Carregar os dados do CSV
#     df = pd.read_csv(csv_file)
    
#     # Armazenar o primeiro timestamp do CSV para sincronização
#     min_csv_time = df['timestamp'].min()
    
#     # Converter timestamp para datetime
#     df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
#     # Ordenar por tempo para garantir que a linha do tempo esteja correta no gráfico
#     df = df.sort_values(by='datetime')

#     # --- INÍCIO DA FILTRAGEM ---
#     # Agrupa por backend e verifica o valor máximo das colunas alvo
#     backend_stats = df.groupby('backend')[['kv_cache_percent', 'running', 'waiting']].max()
    
#     # Mantém apenas os backends onde pelo menos uma dessas colunas é diferente de 0
#     active_backends = backend_stats[(backend_stats != 0).any(axis=1)].index
    
#     # Filtra o dataframe original para manter apenas esses backends ativos
#     df = df[df['backend'].isin(active_backends)]
#     # --- FIM DA FILTRAGEM ---

#     # Obter a lista de backends únicos
#     backends = df['backend'].unique()

#     # --- Processamento dos dados do JSON com Correção de Tempo ---
#     json_times = []
#     active_counts = []
#     if os.path.exists(json_path):
#         with open(json_path, 'r') as file:
#             data = json.load(file)
        
#         programs = data.get("per_program", [])
#         events = []
#         for prog in programs:
#             start_time = prog.get("start_time")
#             end_time = prog.get("end_time")
            
#             if start_time is not None and end_time is not None:
#                 events.append((start_time, 1))
#                 events.append((end_time, -1))
        
#         events.sort(key=lambda x: (x[0], x[1]))
        
#         if events:
#             # --- CORREÇÃO DE SINCRONIZAÇÃO DE TEMPO ---
#             # Se um offset manual não foi fornecido, auto-alinha o primeiro evento do JSON com o primeiro evento do CSV

#             min_json_time = events[0][0]
#             calculated_offset = min_csv_time - min_json_time

#             current_active = 0
            
#             # Estado inicial antes do primeiro evento (aplicando o offset)
#             json_times.append(pd.to_datetime(events[0][0] + calculated_offset - 1, unit='s'))
#             active_counts.append(0)
            
#             for t, change in events:
#                 current_active += change
#                 # Aplica o offset para cada tempo do JSON
#                 json_times.append(pd.to_datetime(t + calculated_offset, unit='s'))
#                 active_counts.append(current_active)
#     else:
#         print(f"Aviso: Arquivo JSON não encontrado em '{json_path}'. O gráfico de Active Programs ficará vazio.")

#     # Criar a figura com 6 subplots (6 linhas, 1 coluna)
#     fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(32, 24), sharex=True)
    
#     # Desempacotar os 6 eixos
#     ax_kv, ax_running, ax_waiting, ax_global, ax_global_running, ax_active_progs = axes

#     # Plotar as métricas por backend
#     for backend in backends:
#         backend_df = df[df['backend'] == backend]
        
#         window_size = 1

#         # Plotando com média móvel
#         ax_kv.plot(backend_df['datetime'], backend_df['kv_cache_percent'].rolling(window_size).mean(), label=backend, linewidth=1.5)
#         ax_running.plot(backend_df['datetime'], backend_df['running'].rolling(window_size).mean(), label=backend, linewidth=1.5)
#         ax_waiting.plot(backend_df['datetime'], backend_df['waiting'].rolling(window_size).mean(), label=backend, linewidth=1.5)
        
#     # Configurações do gráfico de KV Cache Percent
#     ax_kv.set_title('KV Cache Percent por Backend')
#     ax_kv.set_ylabel('KV Cache (%)')
#     ax_kv.grid(True)
    
#     # Configurações do gráfico de Running
#     ax_running.set_title('Running Requests por Backend')
#     ax_running.set_ylabel('Running')
#     ax_running.grid(True)
    
#     # Configurações do gráfico de Waiting
#     ax_waiting.set_title('Waiting Requests por Backend')
#     ax_waiting.set_ylabel('Waiting')
#     ax_waiting.grid(True)

#     # Para as métricas globais
#     global_df = df.groupby('datetime')[['incomplete_count', 'proxy_waiting_count']].max().reset_index()
    
#     # Cálculo: Running requests globais
#     global_df['global_running'] = global_df['incomplete_count'] - global_df['proxy_waiting_count']
    
#     # Plotando métricas globais originais
#     ax_global.plot(global_df['datetime'], global_df['incomplete_count'], label='Incomplete Count', color='red', linewidth=2)
#     ax_global.plot(global_df['datetime'], global_df['proxy_waiting_count'], label='Proxy Waiting Count', color='purple', linewidth=2)
    
#     # Configurações do gráfico global original
#     ax_global.set_title('Métricas Globais (Incomplete & Proxy Waiting)')
#     ax_global.set_ylabel('Count')
#     ax_global.grid(True)
#     ax_global.legend()

#     # Plotando a nova métrica global de Running Requests
#     ax_global_running.plot(global_df['datetime'], global_df['global_running'], label='Running Requests (Incomplete - Waiting)', color='green', linewidth=2)
    
#     # Configurações do gráfico global de running
#     ax_global_running.set_title('Global Running Requests')
#     ax_global_running.set_ylabel('Running')
#     ax_global_running.grid(True)
#     ax_global_running.legend()

#     # Plotando o gráfico de Active Programs do JSON
#     if json_times and active_counts:
#         ax_active_progs.step(json_times, active_counts, where='post', color='blue', linewidth=2, label='Active Programs')
#         ax_active_progs.fill_between(json_times, active_counts, step='post', color='blue', alpha=0.1)

#     ax_active_progs.set_title('Concurrent Active Programs Over Time (JSON) - Time Synced')
#     ax_active_progs.set_xlabel('Tempo')
#     ax_active_progs.set_ylabel('Active Programs')
#     ax_active_progs.grid(True)
#     ax_active_progs.legend(loc='upper right')

#     # Colocar a legenda dos backends fora do gráfico
#     if len(backends) > 0:
#         handles, labels = ax_kv.get_legend_handles_labels()
#         fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), title="Backends")

#     # Adicionar o título geral
#     if title:
#         fig.suptitle(title, fontsize=24, fontweight='bold', y=0.98)

#     # Ajustar o layout
#     plt.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])
    
#     # Mostrar o gráfico
#     plt.show()

import statistics

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
            
            # --- MODIFICAÇÃO AQUI ---
            # Ignora o valor se for a run 1 e o load balancer for EXATAMENTE "autellix"
            if run == 1 and lb == "autellix":
                val = None
            # ------------------------
            
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

def results(folder, y_metric,rate="*"):
    resultados = defaultdict(lambda: defaultdict(dict))

    # padrao_baseline = os.path.join(folder, f"baseline_output_atlas_service_cumulative_*_rate{rate}_run*.json")
    # padrao_normal = os.path.join(folder, f"output_atlas_service_cumulative_*_rate{rate}_run*.json")

    padrao_baseline = os.path.join(folder, f"baseline_output_fcfs_*_rate{rate}_run*.json")
    padrao_normal = os.path.join(folder, f"output_fcfs_*_rate{rate}_run*.json")    
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

import statistics

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
    
    # Variável para armazenar os valores válidos e calcular a média/DP depois
    valores_por_coluna = {col: [] for col in colunas}
    
    # Preenche as linhas das runs
    for run in range(1, max_run + 1):
        linha = f"{run:02d}"
        for col in colunas:
            run_type, lb = col
            val = resultados[run_type][lb].get(run, {}).get("hit_rate")
            
            # --- MODIFICAÇÃO AQUI ---
            # Ignora o valor se for a run 1 e o load balancer for EXATAMENTE "autellix"
            if run == 1 and lb == "autellix":
                val = None
            # ------------------------
            
            if val is not None:
                linha += f" & {val:.4f}"
                valores_por_coluna[col].append(val)
            else:
                linha += " & - "
        linha += r" \\"
        latex.append(linha)
        
    latex.append(r"\hline")
    
    # Linha de resumo: Média \pm Desvio Padrão
    linha_resumo = r"\textbf{Média $\pm$ DP}"
    for col in colunas:
        vals = valores_por_coluna[col] # Usa apenas os valores que passaram pelo filtro
        
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
    # padrao_baseline = os.path.join(folder, f"baseline_output_fcfs_*_rate*_run*.json")
    # padrao_normal = os.path.join(folder, f"output_fcfs_*_rate*_run*.json")   
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

def plot_engine_metrics(json_path: str):
    """
    Reads an output JSON, extracts prefix hit rates and engine request counts,
    and plots them as a grouped bar chart.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    full_metrics = data.get("full_metrics", {})
    prefix_cache = full_metrics.get("prefix_cache", {})
    request_counts = full_metrics.get("engine_request_counts", {})
    
    # Dictionary to align metrics by port number
    engines = {}
    
    # 1. Process Prefix Cache Data
    for engine_name, metrics in prefix_cache.items():
        clean_name = engine_name
        # Ignore the _0 at the end if it exists
        if clean_name.endswith("_0"):
            clean_name = clean_name[:-2] 
            
        # Extract the port (e.g., "8106" from "engine_8106")
        port = clean_name.split('_')[-1] if '_' in clean_name else clean_name
        engines[port] = {'hit_rate': metrics.get("hit_rate", 0), 'requests': 0}
        
    # 2. Process Request Counts Data
    for url, count in request_counts.items():
        # Extract the port from the URL (e.g., "8106" from "http://localhost:8106")
        port = url.split(':')[-1]
        
        # If the engine had requests but wasn't in prefix cache, initialize it
        if port not in engines:
            engines[port] = {'hit_rate': 0, 'requests': 0}
        engines[port]['requests'] = count
        
    # Prepare data for plotting
    ports = sorted(list(engines.keys()))
    hit_rates = [engines[p]['hit_rate'] for p in ports]
    requests = [engines[p]['requests'] for p in ports]
    
    # Set up the plot geometry
    x = np.arange(len(ports))
    width = 0.35  # Width of the bars
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- Axis 1: Number of Requests ---
    color1 = '#1f77b4' # Standard matplotlib blue
    bars1 = ax1.bar(x - width/2, requests, width, label='Number of Requests', color=color1)
    ax1.set_xlabel('Engine Port', fontweight='bold')
    ax1.set_ylabel('Number of Requests', color=color1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ports)
    
    # --- Axis 2: Prefix Hit Rate (Twin X) ---
    ax2 = ax1.twinx()
    color2 = '#ff7f0e' # Standard matplotlib orange
    bars2 = ax2.bar(x + width/2, hit_rates, width, label='Prefix Hit Rate', color=color2)
    ax2.set_ylabel('Hit Rate (0.0 to 1.0)', color=color2, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Pad the top of the hit rate scale slightly so bars don't touch the very top
    max_hit_rate = max(hit_rates) if hit_rates else 1.0
    ax2.set_ylim(0, max(max_hit_rate * 1.15, 1.0))
    
    # Consolidate legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Engine Metrics: Requests vs. Prefix Hit Rate', fontweight='bold', fontsize=14)
    fig.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.show()
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_file, title=None, json_path=None, show_engine_load=False):
    # Inferir o caminho do JSON se não for fornecido
    if json_path is None:
        json_path = csv_file.replace('_kv_cache.csv', '.json')

    # Carregar os dados do CSV
    df = pd.read_csv(csv_file)
    
    # Armazenar o primeiro timestamp do CSV para sincronização
    min_csv_time = df['timestamp'].min()
    
    # Converter timestamp para datetime
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

    # Obter a lista de backends únicos
    backends = df['backend'].unique()

    # --- Processamento dos dados do JSON com Correção de Tempo ---
    json_times = []
    active_counts = []
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        programs = data.get("per_program", [])
        events = []
        for prog in programs:
            start_time = prog.get("start_time")
            end_time = prog.get("end_time")
            
            if start_time is not None and end_time is not None:
                events.append((start_time, 1))
                events.append((end_time, -1))
        
        events.sort(key=lambda x: (x[0], x[1]))
        
        if events:
            # --- CORREÇÃO DE SINCRONIZAÇÃO DE TEMPO ---
            min_json_time = events[0][0]
            calculated_offset = min_csv_time - min_json_time

            current_active = 0
            
            # Estado inicial antes do primeiro evento (aplicando o offset)
            json_times.append(pd.to_datetime(events[0][0] + calculated_offset - 1, unit='s'))
            active_counts.append(0)
            
            for t, change in events:
                current_active += change
                # Aplica o offset para cada tempo do JSON
                json_times.append(pd.to_datetime(t + calculated_offset, unit='s'))
                active_counts.append(current_active)
    else:
        print(f"Aviso: Arquivo JSON não encontrado em '{json_path}'. O gráfico de Active Programs ficará vazio.")

    # Ajustar número de subplots dependendo da flag 'show_engine_load'
    nrows = 5 if show_engine_load else 6
    fig_height = 20 if show_engine_load else 24
    
    # Criar a figura
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(32, fig_height), sharex=True)
    
    # Desempacotar os eixos dinamicamente
    if show_engine_load:
        ax_kv, ax_load, ax_global, ax_global_running, ax_active_progs = axes
    else:
        ax_kv, ax_running, ax_waiting, ax_global, ax_global_running, ax_active_progs = axes

    # Plotar as métricas por backend
    for backend in backends:
        backend_df = df[df['backend'] == backend]
        window_size = 1

        # Plotando KV com média móvel
        ax_kv.plot(backend_df['datetime'], backend_df['kv_cache_percent'].rolling(window_size).mean(), label=backend, linewidth=1.5)
        
        if show_engine_load:
            # Calcular e plotar Engine Load (running + waiting)
            engine_load = backend_df['running'] + backend_df['waiting']
            ax_load.plot(backend_df['datetime'], engine_load.rolling(window_size).mean(), label=backend, linewidth=1.5)
        else:
            # Plotar separadamente
            ax_running.plot(backend_df['datetime'], backend_df['running'].rolling(window_size).mean(), label=backend, linewidth=1.5)
            ax_waiting.plot(backend_df['datetime'], backend_df['waiting'].rolling(window_size).mean(), label=backend, linewidth=1.5)
        
    # Configurações do gráfico de KV Cache Percent
    ax_kv.set_title('KV Cache Percent por Backend')
    ax_kv.set_ylabel('KV Cache (%)')
    ax_kv.grid(True)
    
    # Configurações dinâmicas de Running/Waiting vs Engine Load
    if show_engine_load:
        ax_load.set_title('Engine Load (Running + Waiting) por Backend')
        ax_load.set_ylabel('Total Requests')
        ax_load.grid(True)
    else:
        ax_running.set_title('Running Requests por Backend')
        ax_running.set_ylabel('Running')
        ax_running.grid(True)
        
        ax_waiting.set_title('Waiting Requests por Backend')
        ax_waiting.set_ylabel('Waiting')
        ax_waiting.grid(True)

    # Para as métricas globais
    global_df = df.groupby('datetime')[['incomplete_count', 'proxy_waiting_count']].max().reset_index()
    
    # Cálculo: Running requests globais
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
    
    # Configurações do gráfico global de running
    ax_global_running.set_title('Global Running Requests')
    ax_global_running.set_ylabel('Running')
    ax_global_running.grid(True)
    ax_global_running.legend()

    # Plotando o gráfico de Active Programs do JSON
    if json_times and active_counts:
        ax_active_progs.step(json_times, active_counts, where='post', color='blue', linewidth=2, label='Active Programs')
        ax_active_progs.fill_between(json_times, active_counts, step='post', color='blue', alpha=0.1)

    ax_active_progs.set_title('Concurrent Active Programs Over Time (JSON) - Time Synced')
    ax_active_progs.set_xlabel('Tempo')
    ax_active_progs.set_ylabel('Active Programs')
    ax_active_progs.grid(True)
    ax_active_progs.legend(loc='upper right')

    # --- NOVA LINHA VERTICAL (50 segundos do início) ---
    # line_time = pd.to_datetime(min_csv_time + 0, unit='s')
    # for ax in axes:
    #     ax.axvline(x=line_time, color='black', linestyle='--', linewidth=2, alpha=0.8, label='20s mark' if ax == ax_kv else "")

    # line_time = pd.to_datetime(min_csv_time + 250, unit='s')
    # for ax in axes:
    #     ax.axvline(x=line_time, color='black', linestyle='--', linewidth=2, alpha=0.8, label='270s mark' if ax == ax_kv else "")
        
    # Colocar a legenda dos backends fora do gráfico
    if len(backends) > 0:
        handles, labels = ax_kv.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), title="Backends")

    # Adicionar o título geral
    if title:
        fig.suptitle(title, fontsize=24, fontweight='bold', y=0.98)

    # Ajustar o layout
    plt.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])
    
    # Mostrar o gráfico
    plt.show()

import os
import glob
import json

def recalcular_throughput_nos_jsons(pasta):
    """
    Lê os arquivos JSON, recalcula os throughputs de tokens (input, output e total)
    baseado no tempo 'total_e2el_s' e salva a alteração diretamente no próprio arquivo.
    """
    padrao_busca = os.path.join(pasta, "baseline_output_fcfs_autellix_rateinf_*progs_run1.json")
    arquivos = glob.glob(padrao_busca)
    
    arquivos_atualizados = 0
    
    for arquivo in arquivos:
        nome_arquivo = os.path.basename(arquivo)
        
        # 1. Lê o conteúdo atual do JSON
        with open(arquivo, 'r', encoding='utf-8') as f:
            try:
                conteudo = json.load(f)
            except json.JSONDecodeError:
                print(f"Erro ao ler o arquivo {nome_arquivo}. JSON inválido.")
                continue
                
        # 2. Verifica se a chave 'full_metrics' existe
        if "full_metrics" in conteudo:
            fm = conteudo["full_metrics"]
            chaves_necessarias = ["total_input_tokens", "total_output_tokens", "total_e2el_s"]
            
            # Garante que os dados necessários para o cálculo estão presentes
            if all(k in fm for k in chaves_necessarias):
                in_tokens = fm["total_input_tokens"]
                out_tokens = fm["total_output_tokens"]
                tempo_total = fm["total_e2el_s"]
                
                # Previne erro de divisão por zero
                if tempo_total > 0:
                    # Faz os recálculos
                    in_throughput = in_tokens / tempo_total
                    out_throughput = out_tokens / tempo_total
                    total_throughput = (in_tokens + out_tokens) / tempo_total
                    
                    # Substitui/Adiciona os valores no dicionário
                    conteudo["full_metrics"]["input_token_throughput"] = in_throughput
                    conteudo["full_metrics"]["output_token_throughput"] = out_throughput
                    conteudo["full_metrics"]["total_token_throughput"] = total_throughput
                    
                    # 3. Salva as alterações de volta no arquivo original
                    with open(arquivo, 'w', encoding='utf-8') as f_out:
                        # indent=2 mantém o arquivo legível e formatado bonitinho
                        json.dump(conteudo, f_out, indent=2, ensure_ascii=False)
                        
                    print(f"Atualizado: {nome_arquivo}")
                    arquivos_atualizados += 1
                else:
                    print(f"Ignorado: {nome_arquivo} (total_e2el_s é zero ou negativo)")
            else:
                print(f"Ignorado: {nome_arquivo} (faltam chaves base para o cálculo)")
        else:
            print(f"Ignorado: {nome_arquivo} (sem chave 'full_metrics')")
            
    print(f"\nProcesso concluído! {arquivos_atualizados} arquivos foram atualizados.")