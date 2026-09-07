"""
test_reranker.py — Unit tests for the re-ranking layer.

The real CrossEncoder model is stubbed out (see conftest.py) — these tests
verify OUR sorting/filtering/threshold logic, not the model's judgment
quality (that needs a real model + real queries to evaluate).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


@pytest.fixture
def mock_cross_encoder(monkeypatch):
    """Replace the cached cross-encoder model with a fake that returns
    predetermined scores, so we can test rerank()'s sorting logic in isolation."""
    import reranker

    class FakeCrossEncoder:
        def __init__(self, score_map):
            self.score_map = score_map

        def predict(self, pairs):
            return [self.score_map[text] for _, text in pairs]

    def _set_scores(score_map):
        reranker._model_cache = FakeCrossEncoder(score_map)

    yield _set_scores
    reranker._model_cache = None  # reset after test


def test_rerank_sorts_by_score_descending(mock_cross_encoder):
    from reranker import rerank

    candidates = [
        {"chunk_id": "a", "text": "low relevance", "distance": 0.1},
        {"chunk_id": "b", "text": "high relevance", "distance": 0.5},
        {"chunk_id": "c", "text": "medium relevance", "distance": 0.3},
    ]
    mock_cross_encoder({"low relevance": 1.0, "high relevance": 9.0, "medium relevance": 5.0})

    result = rerank("test query", candidates, top_n=3)

    assert [r["chunk_id"] for r in result] == ["b", "c", "a"]


def test_rerank_respects_top_n_limit(mock_cross_encoder):
    from reranker import rerank

    candidates = [
        {"chunk_id": str(i), "text": f"text {i}", "distance": 0.1} for i in range(10)
    ]
    mock_cross_encoder({f"text {i}": float(i) for i in range(10)})

    result = rerank("test query", candidates, top_n=3)

    assert len(result) == 3
    # Highest scores (9, 8, 7) should be kept
    assert [r["chunk_id"] for r in result] == ["9", "8", "7"]


def test_rerank_handles_empty_candidates(mock_cross_encoder):
    from reranker import rerank

    result = rerank("test query", [], top_n=5)

    assert result == []


def test_rerank_adds_score_field_to_every_candidate(mock_cross_encoder):
    from reranker import rerank

    candidates = [{"chunk_id": "a", "text": "some text", "distance": 0.2}]
    mock_cross_encoder({"some text": 3.5})

    result = rerank("test query", candidates, top_n=1)

    assert "rerank_score" in result[0]
    assert result[0]["rerank_score"] == 3.5


def test_is_confident_true_above_threshold():
    from reranker import is_confident, CONFIDENCE_THRESHOLD

    results = [{"chunk_id": "a", "rerank_score": CONFIDENCE_THRESHOLD + 1.0}]

    assert is_confident(results) is True


def test_is_confident_false_below_threshold():
    from reranker import is_confident, CONFIDENCE_THRESHOLD

    results = [{"chunk_id": "a", "rerank_score": CONFIDENCE_THRESHOLD - 1.0}]

    assert is_confident(results) is False


def test_is_confident_false_for_empty_results():
    from reranker import is_confident

    assert is_confident([]) is False