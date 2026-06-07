export CUSTOM_API_BASE="http://localhost:8081"
export CUSTOM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
python run.py \
  --algorithm lats \
  --task_start_index 900 \
  --task_end_index 902 \
  --iterations 3 \
  --log logs/lats_smoke_test.log \
  --n_generate_sample 10 \
  --n_evaluate_sample 1 \
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
