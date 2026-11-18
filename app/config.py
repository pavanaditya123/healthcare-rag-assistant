"""All settings come from environment variables so the same image runs anywhere."""
import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # Vector store
    qdrant_url: str = _env("QDRANT_URL", "")          # empty -> local on-disk mode
    qdrant_path: str = _env("QDRANT_PATH", "qdrant_storage")
    collection: str = _env("QDRANT_COLLECTION", "medquad")

    # Models (all run on CPU via fastembed / ONNX)
    dense_model: str = _env("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    sparse_model: str = _env("SPARSE_MODEL", "Qdrant/bm25")
    rerank_model: str = _env("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

    # Retrieval knobs
    candidates_per_retriever: int = int(_env("CANDIDATES", "20"))
    top_k: int = int(_env("TOP_K", "5"))
    # Characters of each passage the cross-encoder reads (speed vs accuracy, see evals)
    rerank_max_chars: int = int(_env("RERANK_MAX_CHARS", "1000"))
    # Below this cross-encoder score we say "no reliable info" (tuned in evals/)
    min_rerank_score: float = float(_env("MIN_RERANK_SCORE", "0.0"))

    # LLM (Groq, OpenAI-compatible API)
    llm_provider: str = _env("LLM_PROVIDER", "groq")   # "groq" or "stub"
    groq_api_key: str = _env("GROQ_API_KEY", "")
    groq_url: str = _env("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
    primary_model: str = _env("LLM_PRIMARY_MODEL", "llama-3.3-70b-versatile")
    fallback_model: str = _env("LLM_FALLBACK_MODEL", "llama-3.1-8b-instant")
    llm_timeout_s: float = float(_env("LLM_TIMEOUT_S", "10"))
    llm_retries: int = int(_env("LLM_RETRIES", "1"))
    llm_max_tokens: int = int(_env("LLM_MAX_TOKENS", "300"))  # ~150-word answers; fewer tokens = faster

    # Redis cache + rate limit
    redis_url: str = _env("REDIS_URL", "")             # empty -> in-process fallback
    cache_ttl_s: int = int(_env("CACHE_TTL_S", "3600"))
    rate_limit_per_min: int = int(_env("RATE_LIMIT_PER_MIN", "30"))


settings = Settings()
