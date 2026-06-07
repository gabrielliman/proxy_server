# utils/tokenizer_utils.py
from functools import lru_cache
from transformers import AutoTokenizer
from config.settings import MODEL

@lru_cache(maxsize=1)
def get_tokenizer():
    """Retorna uma instância única do tokenizer (Singleton)"""
    print(f"Carregando tokenizer para o modelo: {MODEL}")
    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)