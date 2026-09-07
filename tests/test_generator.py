"""
test_generator.py — Unit tests for the generation layer.

The real Groq client is stubbed (see conftest.py + local fakes below) —
these tests verify OUR logic: context formatting, source deduplication,
insufficient-context detection, and retry/backoff behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def test_build_context_block_labels_sources_sequentially():
    from generator import build_context_block

    chunks = [
        {"source_url": "https://example.com/a", "text": "First chunk text"},
        {"source_url": "https://example.com/b", "text": "Second chunk text"},
    ]

    result = build_context_block(chunks)

    assert "[Source 1: https://example.com/a]" in result
    assert "[Source 2: https://example.com/b]" in result
    assert "First chunk text" in result
    assert "Second chunk text" in result


def test_build_context_block_handles_empty_chunks():
    from generator import build_context_block

    result = build_context_block([])

    assert result == "No context available."


def test_generate_answer_deduplicates_sources(monkeypatch):
    """Two chunks from the same URL should collapse to one source in the output."""
    import generator as g

    class FakeMessage:
        content = "Answer text [Source: https://example.com/it]"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(g, "get_groq_client", lambda: FakeClient())

    chunks = [
        {"chunk_id": "a", "text": "text a", "source_url": "https://example.com/it", "source_title": "IT"},
        {"chunk_id": "b", "text": "text b", "source_url": "https://example.com/it", "source_title": "IT"},
    ]

    result = g.generate_answer("some question", chunks)

    assert result["sources"] == ["https://example.com/it"]
    assert result["insufficient"] is False


def test_generate_answer_detects_insufficient_context(monkeypatch):
    import generator as g

    class FakeMessage:
        content = g.INSUFFICIENT_MARKER

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(g, "get_groq_client", lambda: FakeClient())

    chunks = [{"chunk_id": "a", "text": "unrelated", "source_url": "https://x.com", "source_title": "X"}]
    result = g.generate_answer("unanswerable question", chunks)

    assert result["insufficient"] is True
    assert result["sources"] == []


def test_call_groq_with_retry_succeeds_after_transient_failures(monkeypatch):
    import generator as g

    monkeypatch.setattr(g, "RETRY_BACKOFF_SECONDS", 0.001)  # skip real sleep in tests

    call_count = {"n": 0}

    class FlakyCompletions:
        def create(self, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise Exception("rate limit exceeded")

            class FakeMessage:
                content = "Success"

            class FakeChoice:
                message = FakeMessage()

            class FakeResponse:
                choices = [FakeChoice()]

            return FakeResponse()

    class FakeChat:
        completions = FlakyCompletions()

    class FakeClient:
        chat = FakeChat()

    result = g._call_groq_with_retry(FakeClient(), [{"role": "user", "content": "hi"}])

    assert result == "Success"
    assert call_count["n"] == 3


def test_call_groq_with_retry_fails_fast_on_auth_error(monkeypatch):
    from exceptions import GenerationError
    import generator as g

    call_count = {"n": 0}

    class AuthFailCompletions:
        def create(self, **kwargs):
            call_count["n"] += 1
            raise Exception("401 invalid api key")

    class FakeChat:
        completions = AuthFailCompletions()

    class FakeClient:
        chat = FakeChat()

    with pytest.raises(GenerationError) as exc_info:
        g._call_groq_with_retry(FakeClient(), [{"role": "user", "content": "hi"}])

    assert exc_info.value.retryable is False
    assert call_count["n"] == 1  # must NOT retry auth failures


def test_get_groq_client_raises_configuration_error_without_api_key(monkeypatch):
    from exceptions import ConfigurationError
    import generator as g

    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ConfigurationError):
        g.get_groq_client()