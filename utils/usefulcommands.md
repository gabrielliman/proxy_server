# Comando para rodar
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dtype bfloat16 \
    --max-model-len 20000 \
    --port 8001 \
    --gpu-memory-utilization 0.6

CUDA_VISIBLE_DEVICES -> qual GPU usar

# roda o biodex em paralelo
python multi_biodex.py --n_prog 3 --n_samples 20 20 20





cd /home/nunes/Abacus/palimpzest/abacus-research
./start_vllm_servers.sh
isso inicia o Qwen3-4B tna porta 8106

cd /scratch/global/proxy_server
conda activate /scratch/global/abacus
python main.py

pra iniciar os múltiplos biodex
cd /home/nunes/Abacus/palimpzest/abacus-research
conda activate /scratch/global/abacus
python multi_biodex.py --n_prog 3 --n_samples 10 30 50 --prog_ids prog_10 prog_30 prog_50


benchmark sharegpt multiplos programas:
python benchmark_sharegpt.py --base-url http://localhost:8080 --dataset /scratch/global/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --limit 100 --model Qwen/Qwen3-4B

em outro terminal:
curl localhost:8080/processes | jq .
pra retornar a tabela de processos no momento da execução