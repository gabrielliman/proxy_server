# utils/tokenizer_utils.py
from functools import lru_cache
from transformers import AutoTokenizer
from config.settings import MODEL

@lru_cache(maxsize=1)
def get_tokenizer():
    """Retorna uma instância única do tokenizer (Singleton)"""
    print(f"Carregando tokenizer para o modelo: {MODEL}")
    try:
        # Attempt to load the primary model's tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, model_max_length=10000)
        
    except Exception as e:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2",trust_remote_code=True, model_max_length=10000)
    return tokenizer