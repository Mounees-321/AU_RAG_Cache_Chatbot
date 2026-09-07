"""
semantic_cache.py — Semantic Caching Layer (Redis + RedisVL)

This module sits IN FRONT of the RAG pipeline (retrieval -> rerank ->
generation). router.py calls check() before doing any retrieval work at
all, and store() after a fresh answer has been produced.

Flow this module enables:

    User Query -> semantic_cache.check()
        Cache HIT  -> return cached answer immediately.
                      Vector store, cross-encoder, and the LLM are never
                      touched -> zero retrieval cost, zero LLM cost.
        Cache MISS -> caller runs the normal RAG + LLM pipeline, then
                      calls semantic_cache.store() to save the answer for
                      the next semantically-similar question.

Design goals:
    - Fail open, always. If Redis is unreachable, misconfigured, or the
      redis/redisvl packages aren't installed, every check() silently
      returns None (a "miss") and every store() is a silent no-op. The
      RAG + LLM pipeline must keep working exactly as before with the
      cache effectively turned off. Callers never need to special-case
      "is the cache up?" — they just call check()/store() and get a
      well-defined result either way.
    - No hard-coded secrets or endpoints. Every tunable (Redis connection
      string, similarity threshold, TTL, cache name, on/off switch) comes
      from environment variables, loaded via python-dotenv from a local
      .env file (see .env.example).
    - Reuse infrastructure already in the project. The embedding model
      used for the cache's vector similarity is the same free, local
      sentence-transformers model already used elsewhere in the pipeline
      style (small, fast, no extra API key), so there's no second
      embedding provider to configure.

Env vars (see .env.example for a documented template):
    REDIS_URL                    default: redis://localhost:6379
    SEMANTIC_CACHE_ENABLED       default: true   (set to "false" to disable
                                                   the cache entirely, e.g.
                                                   for local debugging)
    SEMANTIC_CACHE_THRESHOLD     default: 0.15   (cosine distance cutoff —
                                                   LOWER = stricter match;
                                                   0 = exact only, ~0.2-0.3
                                                   = loosely similar)
    SEMANTIC_CACHE_TTL_SECONDS   default: 86400  (24h — how long a cached
                                                   answer stays valid)
    SEMANTIC_CACHE_NAME          default: au_it_rag_cache (Redis index name)
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---- Config (all overridable via .env / real env vars) --------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

_enabled_raw = os.getenv("SEMANTIC_CACHE_ENABLED", "true").strip().lower()
CACHE_ENABLED = _enabled_raw not in ("false", "0", "no", "off")

DISTANCE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.15"))
TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", str(24 * 60 * 60)))
CACHE_NAME = os.getenv("SEMANTIC_CACHE_NAME", "au_it_rag_cache")

# Free, local embedding model — no API key, no extra cost. Independent of
# ChromaDB's own embedding function in vectorstore.py; only used to embed
# cache keys (questions), not document chunks.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- Lazy, fail-open initialization ---------------------------------------
# We don't connect to Redis at import time: importing this module must
# never crash the app just because Redis happens to be down. Connection is
# attempted once, lazily, on first use.

_cache = None          # the redisvl SemanticCache instance, once connected
_init_attempted = False
_available = False     # True only if Redis + cache initialized successfully

def _normalize(question: str) -> str:
    """
    Strip incidental formatting differences (whitespace, casing, trailing
    punctuation) before embedding, so the cache's distance calculation is
    spent on real wording differences, not "?" vs "" or a trailing newline.
    """
    return question.strip().rstrip("?.! \n\t").lower()

def _try_init() -> None:
    global _cache, _init_attempted, _available
    if _init_attempted:
        return
    _init_attempted = True

    if not CACHE_ENABLED:
        logger.info("Semantic cache disabled via SEMANTIC_CACHE_ENABLED=false")
        return

    try:
        import redis
        # pyrefly: ignore [missing-import]
        from redisvl.extensions.cache.llm import SemanticCache
        # pyrefly: ignore [missing-import]
        from redisvl.utils.vectorize import HFTextVectorizer

        client = redis.Redis.from_url(
            REDIS_URL, socket_connect_timeout=2, socket_timeout=2
        )
        client.ping()  # fail fast here if Redis isn't reachable

        vectorizer = HFTextVectorizer(model=EMBEDDING_MODEL)
        cache = SemanticCache(
            name=CACHE_NAME,
            vectorizer=vectorizer,
            redis_client=client,
            distance_threshold=DISTANCE_THRESHOLD,
        )
        cache.set_ttl(TTL_SECONDS)

        _cache = cache
        _available = True
        logger.info(
            f"Semantic cache connected -> {REDIS_URL} "
            f"(index={CACHE_NAME}, threshold={DISTANCE_THRESHOLD}, ttl={TTL_SECONDS}s)"
        )

    except Exception as e:
        # Covers: redis/redisvl not installed, connection refused, auth
        # failure, DNS failure, model download failure, etc. Whatever the
        # cause, the pipeline must still work without the cache.
        _cache = None
        _available = False
        logger.warning(
            f"Semantic cache unavailable ({e!r}) — continuing without it; "
            f"every query will go through the normal RAG + LLM pipeline."
        )


def is_available() -> bool:
    """True if the cache is enabled AND Redis is reachable right now."""
    _try_init()
    return _available


# ---- Public API -------------------------------------------------------

def check(question: str) -> dict | None:
    """
    Look up a semantically similar previously-answered question.

    Returns:
        None                — cache miss, OR cache disabled/unavailable.
                               Either way, the caller should proceed with
                               the normal RAG + LLM pipeline.
        {                   — cache hit
            "answer": str,
            "sources": list[str],
            "matched_question": str,   # the cached prompt that matched
            "distance": float,         # cosine distance (lower = closer)
        }
    """
    _try_init()
    if not _available:
        return None

    normalized = _normalize(question)

    try:
        results = _cache.check(normalized)
        if not results:
            # Diagnostic: see how close the nearest miss actually was
            debug_results = _cache.check(normalized, distance_threshold=1.0)
            if debug_results:
                logger.info(
                    f"CACHE MISS: {question!r} — nearest was "
                    f"{debug_results[0]['prompt']!r} at distance="
                    f"{debug_results[0]['vector_distance']}"
                )
            else:
                logger.info(f"CACHE MISS: {question!r} — no candidates in cache")
            return None
    except Exception as e:
        logger.warning(f"Semantic cache lookup failed ({e!r}) — treating as a miss")
        return None

    top = results[0]
    matched_question = top.get("prompt")
    distance = top.get("vector_distance")
    logger.info(
        f"CACHE HIT: {question!r} matched cached question "
        f"{matched_question!r} (distance={distance})"
    )

    return {
        "answer": top.get("response"),
        "sources": _decode_sources(top.get("metadata")),
        "matched_question": matched_question,
        "distance": distance,
    }


def store(question: str, answer: str, sources: list[str] | None = None) -> None:
    """
    Cache a freshly generated answer so future semantically similar
    questions can be served instantly. Silently does nothing if the cache
    is disabled/unavailable — callers don't need to check first.
    """
    _try_init()
    if not _available:
        return

    if not answer:
        return  # nothing useful to cache

    try:
        _cache.store(
            prompt=_normalize(question),
            response=answer,
            metadata={"sources": sources or []},
        )
        logger.info(f"CACHE STORE: {question!r}")
    except Exception as e:
        logger.warning(f"Failed to store response in semantic cache ({e!r})")


def clear() -> None:
    """Flush all cached entries. Useful for tests or a manual cache reset."""
    _try_init()
    if not _available:
        return
    try:
        _cache.clear()
        logger.info("Semantic cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear semantic cache ({e!r})")


def _decode_sources(metadata) -> list[str]:
    if isinstance(metadata, dict):
        sources = metadata.get("sources", [])
        if isinstance(sources, list):
            return sources
    return []
