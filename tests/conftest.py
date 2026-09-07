"""
conftest.py — Shared pytest fixtures

Stubs out heavy external dependencies (sentence-transformers, chromadb,
groq, ddgs) so the test suite runs fast, offline, and without needing
real API keys or downloaded models. This is standard practice: unit tests
should test YOUR logic, not re-verify that third-party libraries work.

pytest automatically discovers this file and applies these fixtures to
every test in the tests/ directory.
"""

import sys
import types

import pytest


@pytest.fixture(autouse=True)
def stub_external_dependencies(monkeypatch):
    """
    Runs automatically before every test. Replaces heavy/network-dependent
    imports with lightweight fakes so tests are fast and deterministic.
    """
    fake_modules = {
        "sentence_transformers": types.ModuleType("sentence_transformers"),
        "chromadb": types.ModuleType("chromadb"),
        "chromadb.utils": types.ModuleType("chromadb.utils"),
        "groq": types.ModuleType("groq"),
        "ddgs": types.ModuleType("ddgs"),
        "dotenv": types.ModuleType("dotenv"),
    }

    fake_modules["sentence_transformers"].CrossEncoder = type(
        "FakeCrossEncoder", (), {"__init__": lambda self, *a, **k: None}
    )
    fake_modules["chromadb"].PersistentClient = lambda *a, **k: None
    fake_modules["chromadb.utils"].embedding_functions = types.SimpleNamespace(
        SentenceTransformerEmbeddingFunction=lambda *a, **k: None
    )
    fake_modules["groq"].Groq = type(
        "FakeGroq", (), {"__init__": lambda self, *a, **k: None}
    )
    fake_modules["dotenv"].load_dotenv = lambda *a, **k: None

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, max_results=5):
            return []

    fake_modules["ddgs"].DDGS = FakeDDGS

    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    yield