"""
test_main.py — Tests for the FastAPI REST layer.

Uses FastAPI's TestClient, which calls the app in-process (no real server,
no real network) — fast and standard practice for API testing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    import main
    import router as router_module

    monkeypatch.setattr(
        router_module,
        "route_and_answer",
        lambda q, k1=20, k2=5: {
            "answer": "The IT department offers courses in AI and ML.",
            "sources": ["https://departments.auegov.ac.in/it"],
            "used_search": False,
            "kb_attempted": True,
        },
    )
    return TestClient(main.app)


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_returns_valid_response(client):
    response = client.post("/query", json={"question": "What courses does IT offer?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The IT department offers courses in AI and ML."
    assert data["sources"] == ["https://departments.auegov.ac.in/it"]
    assert data["used_search"] is False
    assert "response_time_seconds" in data


def test_query_rejects_too_short_question(client):
    """Pydantic validation should reject questions under min_length before
    they ever reach the pipeline."""
    response = client.post("/query", json={"question": "hi"})

    assert response.status_code == 422  # FastAPI validation error


def test_query_rejects_missing_question_field(client):
    response = client.post("/query", json={})

    assert response.status_code == 422


def test_query_returns_500_on_pipeline_exception(monkeypatch):
    """
    If the pipeline throws an unexpected error, the API should return a
    clean 500 with a generic message — never leak a raw stack trace to
    the client.
    """
    import main
    import router as router_module

    def broken_route(q, k1=20, k2=5):
        raise RuntimeError("something broke deep in the pipeline")

    monkeypatch.setattr(router_module, "route_and_answer", broken_route)
    test_client = TestClient(main.app)

    response = test_client.post("/query", json={"question": "any valid question here"})

    assert response.status_code == 500
    assert "stack trace" not in response.json()["detail"].lower()
    assert "unexpected error" in response.json()["detail"].lower()


def test_query_returns_500_on_configuration_error(monkeypatch):
    """Missing API key etc. should surface as a clear 500, not a crash."""
    import main
    import router as router_module
    from exceptions import ConfigurationError

    def missing_config(q, k1=20, k2=5):
        raise ConfigurationError("GROQ_API_KEY not found")

    monkeypatch.setattr(router_module, "route_and_answer", missing_config)
    test_client = TestClient(main.app)

    response = test_client.post("/query", json={"question": "any valid question here"})

    assert response.status_code == 500
    assert "configuration" in response.json()["detail"].lower()