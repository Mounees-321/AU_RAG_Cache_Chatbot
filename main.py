"""
main.py — Phase 7: REST API Layer

Wraps the full pipeline (retrieve -> rerank -> generate -> route) in a
FastAPI app with a single POST /query endpoint. This is what turns the
project from "a script you run" into "a system with an interface" that
other applications (a frontend, a Slack bot, curl) could call.

Usage:
    uvicorn main:app --reload
    # then visit http://127.0.0.1:8000/docs for interactive API docs

Example request:
    curl -X POST http://127.0.0.1:8000/query \\
         -H "Content-Type: application/json" \\
         -d '{"question": "What courses does the IT department offer?"}'
"""

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import router
from exceptions import ConfigurationError
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Anna University IT Dept — Agentic RAG API",
    description="Ask questions about the Anna University IT department. "
                "Falls back to live web search when the knowledge base is insufficient.",
    version="1.0.0",
)

# CORS: allows a browser-based frontend on a different origin to call this
# API. Locked to localhost for local dev — widen only if you deploy this
# somewhere and know which frontend origin needs access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---- Request / Response schemas -------------------------------------------
# Pydantic models double as automatic request validation AND as the API
# documentation FastAPI generates at /docs — this is the "API-first
# approach to documentation" a JD like Trimble's would be looking for.

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to ask about the Anna University IT department",
        examples=["What courses does the IT department offer?"],
    )


class QueryResponse(BaseModel):
    answer: str = Field(description="The generated answer, grounded in retrieved context")
    sources: list[str] = Field(description="Source URLs the answer was grounded in")
    used_search: bool = Field(description="True if the answer came from live web search fallback")
    cache_hit: bool = Field(
        default=False,
        description="True if this answer was served from the semantic cache "
                     "without calling the LLM (or the RAG pipeline) at all",
    )
    response_time_seconds: float = Field(description="Time taken to process this request")


class HealthResponse(BaseModel):
    status: str
    service: str


# ---- Routes -------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Basic liveness check — useful for uptime monitoring or load balancers."""
    return {"status": "ok", "service": "au-it-rag-api"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Ask a question. The agent retrieves from the knowledge base first,
    and automatically falls back to live web search if the knowledge
    base doesn't have enough context to answer confidently.
    """
    start_time = time.time()
    logger.info(f"Received query: {request.question!r}")

    try:
        result = router.route_and_answer(request.question)

    except ConfigurationError as e:
        # Server misconfiguration (e.g. missing API key) — this is a 500,
        # not something the client did wrong.
        logger.error(f"Configuration error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error. Please contact the administrator.",
        )

    except Exception as e:
        # Catch-all safety net: never let a raw exception/stack trace leak
        # to the client. Log the real error for debugging, return a clean
        # generic message instead.
        logger.error(f"Unexpected error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your question.",
        )

    elapsed = time.time() - start_time
    logger.info(
        f"Query processed in {elapsed:.2f}s "
        f"(cache_hit={result.get('cache_hit', False)}, used_search={result['used_search']})"
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        used_search=result["used_search"],
        cache_hit=result.get("cache_hit", False),
        response_time_seconds=round(elapsed, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)