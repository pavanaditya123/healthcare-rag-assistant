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

    # CPU protection. Model inference is CPU-bound: running more requests at once
    # than we have cores makes EVERY request slow. So each process lets at most
    # MAX_INFLIGHT searches run; others wait up to QUEUE_TIMEOUT_S, then get a
    # fast 503 (load shedding) and the HPA adds pods.
    max_inflight: int = int(_env("MAX_INFLIGHT", "2"))
    queue_timeout_s: float = float(_env("QUEUE_TIMEOUT_S", "5"))
    onnx_threads: int = int(_env("ONNX_THREADS", "0"))  # 0 = let ONNX decide

    # Retrieval knobs
    # 10 vs 20: same recall@5 (0.926 vs 0.929, 300 MedQuAD questions), rerank 32% faster
    candidates_per_retriever: int = int(_env("CANDIDATES", "10"))
    top_k: int = int(_env("TOP_K", "5"))
    # Characters of each passage the cross-encoder reads (speed vs accuracy, see evals)
    rerank_max_chars: int = int(_env("RERANK_MAX_CHARS", "1000"))
    # Below this cross-encoder score we say "no reliable info" (tuned in evals/)
    min_rerank_score: float = float(_env("MIN_RERANK_SCORE", "0.0"))
    # Output guardrail: share of a cited sentence's content words that must appear in
    # its cited passages (see guardrails.ungrounded).
    grounding_min_support: float = float(_env("GROUNDING_MIN_SUPPORT", "0.5"))
    # Shadow mode by default: score and count every answer, but only BLOCK once the
    # threshold is calibrated against the LLM-as-judge (evals/judge_eval.py). Real
    # gpt-oss answers paraphrase, and 0.5 blocked good answers in the smoke test.
    grounding_enforce: bool = _env("GROUNDING_ENFORCE", "false").lower() == "true"

    # LLM (Groq, OpenAI-compatible API)
    llm_provider: str = _env("LLM_PROVIDER", "groq")   # "groq" or "stub"
    groq_api_key: str = _env("GROQ_API_KEY", "")
    groq_url: str = _env("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
    # Groq retired its Llama chat models (404 in scripts/llm_smoke.py, Nov 2026).
    primary_model: str = _env("LLM_PRIMARY_MODEL", "openai/gpt-oss-120b")
    fallback_model: str = _env("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b")
    llm_timeout_s: float = float(_env("LLM_TIMEOUT_S", "10"))
    llm_retries: int = int(_env("LLM_RETRIES", "1"))
    # gpt-oss is a reasoning model: hidden reasoning tokens count against this limit,
    # so it is larger than a ~150-word answer needs. Low effort keeps reasoning short.
    llm_max_tokens: int = int(_env("LLM_MAX_TOKENS", "1024"))
    llm_reasoning_effort: str = _env("LLM_REASONING_EFFORT", "low")  # gpt-oss only; "" to omit

    # Redis cache + rate limit
    redis_url: str = _env("REDIS_URL", "")             # empty -> in-process fallback
    cache_ttl_s: int = int(_env("CACHE_TTL_S", "3600"))            # final answers
    # Retrieved + reranked passages. Retrieval is ~98% of search time, so caching it
    # separately means an expired answer or an LLM retry costs only the LLM call.
    # Kept longer than answers: passages only change on re-index, and ingest clears them.
    context_cache_ttl_s: int = int(_env("CONTEXT_CACHE_TTL_S", "86400"))
    rate_limit_per_min: int = int(_env("RATE_LIMIT_PER_MIN", "30"))


settings = Settings()
