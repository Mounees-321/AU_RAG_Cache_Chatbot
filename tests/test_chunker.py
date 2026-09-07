"""
test_chunker.py — Unit tests for the chunking layer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from chunker import chunk_page, make_splitter, MIN_CHUNK_CHARS


@pytest.fixture
def splitter():
    return make_splitter()


def test_chunk_page_splits_long_text(splitter):
    """A page with long text should be split into more than one chunk."""
    long_text = "This is a sentence about the IT department. " * 50  # ~2300 chars
    page = {"url": "https://example.com/it", "title": "IT Dept", "text": long_text}

    chunks = chunk_page(page, splitter)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["source_url"] == "https://example.com/it"
        assert chunk["source_title"] == "IT Dept"


def test_chunk_page_preserves_short_text_as_single_chunk(splitter):
    """Text shorter than chunk_size should come back as exactly one chunk."""
    short_text = "The IT department has 16 faculty members."
    page = {"url": "https://example.com/it", "title": "IT Dept", "text": short_text}

    chunks = chunk_page(page, splitter)

    assert len(chunks) == 1
    assert chunks[0]["text"] == short_text


def test_chunk_page_handles_empty_text(splitter):
    """Empty text should return an empty list, not crash."""
    page = {"url": "https://example.com/empty", "title": "Empty", "text": ""}

    chunks = chunk_page(page, splitter)

    assert chunks == []


def test_chunk_page_drops_tiny_fragments(splitter):
    """
    Chunks smaller than MIN_CHUNK_CHARS should be filtered out — these are
    usually leftover fragments (e.g. a stray word after splitting), not
    useful retrieval units.
    """
    text = "Hi. " + ("Real content about IT courses and programs. " * 20)
    page = {"url": "https://example.com/it", "title": "IT Dept", "text": text}

    chunks = chunk_page(page, splitter)

    for chunk in chunks:
        assert len(chunk["text"]) >= MIN_CHUNK_CHARS


def test_chunk_ids_are_unique_within_a_page(splitter):
    """Each chunk from the same page must have a distinct chunk_id."""
    long_text = "Course information for the department. " * 60
    page = {"url": "https://example.com/it/courses", "title": "Courses", "text": long_text}

    chunks = chunk_page(page, splitter)
    chunk_ids = [c["chunk_id"] for c in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))  # no duplicates


def test_chunk_index_increments_in_order(splitter):
    """chunk_index should be sequential (0, 1, 2, ...) matching document order."""
    long_text = "Sentence about faculty and research. " * 60
    page = {"url": "https://example.com/it/teaching", "title": "Teaching", "text": long_text}

    chunks = chunk_page(page, splitter)
    indices = [c["chunk_index"] for c in chunks]

    assert indices == list(range(len(chunks)))