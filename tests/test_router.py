"""
test_router.py — Integration tests for the agentic routing layer.

Unlike test_chunker/test_reranker/test_generator (which test one function
at a time), these tests verify how generator.py and router.py work
TOGETHER — this is what actually validates the "autonomous routing
mechanism" resume claim, since routing only makes sense as an end-to-end
behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def test_router_answers_from_kb_when_context_is_sufficient(monkeypatch):
    import router
    import generator

    monkeypatch.setattr(
        generator,
        "answer_question",
        lambda q, k1=20, k2=5: {
            "answer": "The IT department offers AI/ML courses.",
            "sources": ["https://au.edu/it"],
            "insufficient": False,
        },
    )

    result = router.route_and_answer("What courses does IT offer?")

    assert result["used_search"] is False
    assert result["sources"] == ["https://au.edu/it"]


def test_router_falls_back_to_search_when_kb_insufficient(monkeypatch):
    import router
    import generator

    monkeypatch.setattr(
        generator,
        "answer_question",
        lambda q, k1=20, k2=5: {"answer": "INSUFFICIENT_CONTEXT", "sources": [], "insufficient": True},
    )
    monkeypatch.setattr(
        generator,
        "generate_answer",
        lambda q, chunks: {
            "answer": f"Answer from {len(chunks)} search results",
            "sources": [c["source_url"] for c in chunks],
            "insufficient": False,
        },
    )
    monkeypatch.setattr(
        router,
        "web_search",
        lambda q, max_results=5: [
            {"chunk_id": "s1", "text": "search result text", "source_url": "https://live-result.com", "source_title": "Live"}
        ],
    )

    result = router.route_and_answer("What's the weather today?")

    assert result["used_search"] is True
    assert result["sources"] == ["https://live-result.com"]


def test_router_degrades_gracefully_when_both_kb_and_search_fail(monkeypatch):
    import router
    import generator

    monkeypatch.setattr(
        generator,
        "answer_question",
        lambda q, k1=20, k2=5: {"answer": "INSUFFICIENT_CONTEXT", "sources": [], "insufficient": True},
    )
    monkeypatch.setattr(router, "web_search", lambda q, max_results=5: [])

    result = router.route_and_answer("some obscure unanswerable question")

    assert result["used_search"] is True
    assert result["sources"] == []
    assert "try rephrasing" in result["answer"].lower()


def test_router_survives_search_fallback_exception(monkeypatch):
    """
    If the search API itself throws (network down, quota exceeded), routing
    should still return a clean response — never let a raw exception escape
    to whoever is asking the question.
    """
    import router
    import generator
    from exceptions import SearchFallbackError

    monkeypatch.setattr(
        generator,
        "answer_question",
        lambda q, k1=20, k2=5: {"answer": "INSUFFICIENT_CONTEXT", "sources": [], "insufficient": True},
    )

    def broken_search(q, max_results=5):
        raise SearchFallbackError(q, "DDGS timed out")

    monkeypatch.setattr(router, "web_search", broken_search)

    result = router.route_and_answer("query that triggers a broken search")

    assert result["used_search"] is True
    assert result["sources"] == []
    assert "answer" in result  # got a real response, not a crash


def test_router_returns_cached_answer_without_calling_generator(monkeypatch):
    """
    A cache hit must be a genuine pipeline bypass: generator.answer_question
    (retrieval + re-ranking + the LLM call) must never be invoked.
    """
    import router
    import generator
    import semantic_cache

    monkeypatch.setattr(
        semantic_cache,
        "check",
        lambda q: {
            "answer": "Cached answer about IT courses.",
            "sources": ["https://au.edu/it"],
            "matched_question": "What courses does IT offer?",
            "distance": 0.0,
        },
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("generator.answer_question must not be called on a cache hit")

    monkeypatch.setattr(generator, "answer_question", fail_if_called)

    result = router.route_and_answer("What courses does the IT department offer?")

    assert result["cache_hit"] is True
    assert result["kb_attempted"] is False
    assert result["used_search"] is False
    assert result["answer"] == "Cached answer about IT courses."
    assert result["sources"] == ["https://au.edu/it"]


def test_router_stores_kb_answer_in_cache_on_a_miss(monkeypatch):
    """On a cache miss, a successful KB answer must be written back to the cache."""
    import router
    import generator
    import semantic_cache

    monkeypatch.setattr(semantic_cache, "check", lambda q: None)

    stored = {}
    monkeypatch.setattr(
        semantic_cache,
        "store",
        lambda question, answer, sources=None: stored.update(
            {"question": question, "answer": answer, "sources": sources}
        ),
    )
    monkeypatch.setattr(
        generator,
        "answer_question",
        lambda q, k1=20, k2=5: {
            "answer": "The IT department offers AI/ML courses.",
            "sources": ["https://au.edu/it"],
            "insufficient": False,
        },
    )

    result = router.route_and_answer("What courses does IT offer?")

    assert result["cache_hit"] is False
    assert stored["answer"] == "The IT department offers AI/ML courses."
    assert stored["sources"] == ["https://au.edu/it"]


def test_router_stores_search_fallback_answer_in_cache_on_a_miss(monkeypatch):
    """A search-fallback answer must also be cached so a repeat is a hit next time."""
    import router
    import generator
    import semantic_cache

    monkeypatch.setattr(semantic_cache, "check", lambda q: None)

    stored = {}
    monkeypatch.setattr(
        semantic_cache,
        "store",
        lambda question, answer, sources=None: stored.update(
            {"question": question, "answer": answer, "sources": sources}
        ),
    )
    monkeypatch.setattr(
        generator,
        "answer_question",
        lambda q, k1=20, k2=5: {"answer": "INSUFFICIENT_CONTEXT", "sources": [], "insufficient": True},
    )
    monkeypatch.setattr(
        generator,
        "generate_answer",
        lambda q, chunks: {
            "answer": f"Answer from {len(chunks)} search results",
            "sources": [c["source_url"] for c in chunks],
            "insufficient": False,
        },
    )
    monkeypatch.setattr(
        router,
        "web_search",
        lambda q, max_results=5: [
            {"chunk_id": "s1", "text": "search result text", "source_url": "https://live-result.com", "source_title": "Live"}
        ],
    )

    result = router.route_and_answer("What's the weather today?")

    assert result["used_search"] is True
    assert result["cache_hit"] is False
    assert stored["answer"] == "Answer from 1 search results"
    assert stored["sources"] == ["https://live-result.com"]


def test_router_does_not_cache_the_no_results_fallback_message(monkeypatch):
    """The generic 'couldn't find this' apology should never be cached as an answer."""
    import router
    import generator
    import semantic_cache

    monkeypatch.setattr(semantic_cache, "check", lambda q: None)

    store_calls = []
    monkeypatch.setattr(semantic_cache, "store", lambda *a, **k: store_calls.append((a, k)))
    monkeypatch.setattr(
        generator,
        "answer_question",
        lambda q, k1=20, k2=5: {"answer": "INSUFFICIENT_CONTEXT", "sources": [], "insufficient": True},
    )
    monkeypatch.setattr(router, "web_search", lambda q, max_results=5: [])

    result = router.route_and_answer("some obscure unanswerable question")

    assert "try rephrasing" in result["answer"].lower()
    assert store_calls == []


def test_router_survives_redis_being_down(monkeypatch):
    """
    Requirement: Redis failure must not break the pipeline. Simulate
    semantic_cache.check()/store() behaving as they would with Redis down
    (check always returns None, store is a no-op) and confirm the normal
    KB pipeline still answers correctly.
    """
    import router
    import generator
    import semantic_cache

    monkeypatch.setattr(semantic_cache, "check", lambda q: None)
    monkeypatch.setattr(semantic_cache, "store", lambda *a, **k: None)
    monkeypatch.setattr(
        generator,
        "answer_question",
        lambda q, k1=20, k2=5: {
            "answer": "The IT department offers AI/ML courses.",
            "sources": ["https://au.edu/it"],
            "insufficient": False,
        },
    )

    result = router.route_and_answer("What courses does IT offer?")

    assert result["cache_hit"] is False
    assert result["answer"] == "The IT department offers AI/ML courses."
    assert result["used_search"] is False


def test_router_handles_vectorstore_error_without_crashing(monkeypatch):
    """
    If the vector store itself fails (e.g. ChromaDB corrupted, disk issue),
    the router should return a clean error message, not propagate a raw
    exception up to the caller.
    """
    import router
    import generator
    from exceptions import VectorStoreError

    def broken_answer_question(q, k1=20, k2=5):
        raise VectorStoreError("query", "database is locked")

    monkeypatch.setattr(generator, "answer_question", broken_answer_question)

    result = router.route_and_answer("any question")

    assert result["used_search"] is False
    assert "error" in result
    assert "went wrong" in result["answer"].lower()