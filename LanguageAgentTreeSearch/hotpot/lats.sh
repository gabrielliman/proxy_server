export CUSTOM_API_BASE="http://localhost:8081"
export CUSTOM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
python LanguageAgentTreeSearch/hotpot/run.py \
  --algorithm lats \
  --task_start_index 900 \
  --task_end_index 901 \
  --iterations 3 \
  --log ./logs_lats/lats_smoke_test.log \
  --n_generate_sample 5 \
  --n_evaluate_sample 1 \
  --program-rate 0.5 \
  --output-json ./teste.json
# python run.py \
#     --backend gpt-3.5-turbo \
#     --task_start_index 0 \
#     --task_end_index 100 \
#     --n_generate_sample 5 \
#     --n_evaluate_sample 1 \
#     --prompt_sample cot \
#     --temperature 1.0 \
#     --iterations 30 \
#     --log logs/tot_10k.log \
#     ${@}
