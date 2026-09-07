"""
exceptions.py — Phase 6: Custom Exception Hierarchy

Defining specific exception types (instead of letting raw requests/groq/
chromadb exceptions bubble up) makes errors easier to catch selectively,
log meaningfully, and handle differently depending on what failed.

Every custom exception here maps to a real failure point in the pipeline:
    IngestionError      -> ingest.py    (scraping failures)
    VectorStoreError     -> vectorstore.py (embedding/DB failures)
    GenerationError       -> generator.py (Groq API failures)
    SearchFallbackError   -> router.py   (DuckDuckGo search failures)
"""


class RagPipelineError(Exception):
    """Base class for all custom exceptions in this project."""
    pass


class IngestionError(RagPipelineError):
    """Raised when a page fails to fetch or parse during ingestion."""
    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"Failed to ingest {url}: {reason}")


class ChunkingError(RagPipelineError):
    """Raised when a page's text can't be chunked (e.g. empty/malformed input)."""
    def __init__(self, source: str, reason: str):
        self.source = source
        self.reason = reason
        super().__init__(f"Failed to chunk {source}: {reason}")


class VectorStoreError(RagPipelineError):
    """Raised when embedding or querying ChromaDB fails."""
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Vector store {operation} failed: {reason}")


class GenerationError(RagPipelineError):
    """Raised when the Groq API call fails or returns an unusable response."""
    def __init__(self, reason: str, retryable: bool = False):
        self.reason = reason
        self.retryable = retryable
        super().__init__(f"Generation failed: {reason}")


class SearchFallbackError(RagPipelineError):
    """Raised when the live web search fallback itself fails."""
    def __init__(self, query: str, reason: str):
        self.query = query
        self.reason = reason
        super().__init__(f"Search fallback failed for {query!r}: {reason}")


class ConfigurationError(RagPipelineError):
    """Raised for missing/invalid configuration, e.g. missing API keys."""
    pass