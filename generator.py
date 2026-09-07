"""
generator.py — Phase 4: Generation Layer

Takes re-ranked chunks (from reranker.py) and a user question, builds a
grounded prompt, and calls Groq's API to generate a citation-backed answer.

Key design choices:
    - Strict system prompt: only answer from provided context, say so
      explicitly if the context is insufficient (this is what powers the
      routing decision in Phase 5 — router.py checks for the
      INSUFFICIENT_MARKER below).
    - Every answer includes source URLs so responses are auditable, not
      just confident-sounding text.
    - Low temperature for factual consistency over creativity.

Usage:
    python generator.py --query "What courses does the IT department offer?"

Requires:
    GROQ_API_KEY set in a .env file (see .env.example)
"""

import argparse
import logging
import os
import time

from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from groq import Groq

import reranker
from exceptions import ConfigurationError, GenerationError
from logging_config import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

# Retry config for transient Groq failures (rate limits, brief network blips)
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # doubles each retry: 2s, 4s, 8s

# ---- Config -------------------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.2   # low = more factual/consistent, less creative
MAX_TOKENS = 600

# The generator emits this exact marker when it can't answer from context.
# router.py (Phase 5) looks for this string to decide whether to fall back
# to live web search.
INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You are a helpful assistant answering questions about the \
Information Technology department at Anna University, using ONLY the context \
provided below.

Rules:
1. Answer strictly using the given context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, \
respond with exactly: {INSUFFICIENT_MARKER}
   Do not guess or make up an answer.
3. When you do answer, cite the source for each claim using the format [Source: <url>].
4. Keep answers concise and directly relevant to the question.
"""


# ---- Core functions -------------------------------------------------------

def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "GROQ_API_KEY not found. Create a .env file with GROQ_API_KEY=your_key_here "
            "(get a free key at https://console.groq.com/keys)"
        )
    return Groq(api_key=api_key)


def _call_groq_with_retry(client: Groq, messages: list[dict]) -> str:
    """
    Call Groq's chat completion, retrying with exponential backoff on
    transient failures (rate limits, timeouts). Fails fast on errors that
    a retry won't fix (e.g. bad API key -> 401).
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Non-retryable: bad credentials, invalid request — retrying won't help
            if "401" in error_str or "authentication" in error_str or "invalid api key" in error_str:
                raise GenerationError(f"Authentication failed — check your GROQ_API_KEY: {e}", retryable=False)

            # Retryable: rate limits, timeouts, transient network errors
            is_last_attempt = attempt == MAX_RETRIES
            if is_last_attempt:
                break

            wait_time = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                f"Groq call failed (attempt {attempt}/{MAX_RETRIES}): {e}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)

    raise GenerationError(f"Groq call failed after {MAX_RETRIES} attempts: {last_error}", retryable=True)


def build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a labeled context block the LLM can cite
    from directly (e.g. "[Source 1]", "[Source 2]").
    """
    if not chunks:
        return "No context available."

    blocks = []
    for i, chunk in enumerate(chunks, 1):
        blocks.append(
            f"[Source {i}: {chunk['source_url']}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """
    Call Groq to generate an answer grounded in the given chunks.

    Returns:
        {
            "answer": str,
            "sources": list[str],       # unique source URLs actually provided as context
            "insufficient": bool,        # True if the model couldn't answer from context
        }
    """
    client = get_groq_client()
    context_block = build_context_block(chunks)

    user_prompt = f"""Context:
{context_block}

Question: {question}

Answer using only the context above."""

    logger.info(f"Calling Groq ({GROQ_MODEL}) for generation...")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        answer_text = _call_groq_with_retry(client, messages)

    except ConfigurationError:
        # Missing/bad API key — nothing to retry, surface clearly and stop
        raise

    except GenerationError as e:
        # Exhausted retries or hit a non-retryable error — degrade gracefully
        # instead of crashing the whole pipeline.
        logger.error(f"Generation failed permanently: {e}")
        return {
            "answer": "Sorry, I ran into an error generating a response. Please try again in a moment.",
            "sources": [],
            "insufficient": True,
            "error": str(e),
        }

    insufficient = INSUFFICIENT_MARKER in answer_text
    sources = list({c["source_url"] for c in chunks}) if not insufficient else []

    return {
        "answer": answer_text,
        "sources": sources,
        "insufficient": insufficient,
    }


def answer_question(question: str, retrieve_k: int = 20, rerank_n: int = 5) -> dict:
    """
    Full pipeline so far: retrieve -> rerank -> generate.
    This is the function router.py (Phase 5) will wrap with the search fallback.
    """
    chunks = reranker.retrieve_and_rerank(question, retrieve_k, rerank_n)
    return generate_answer(question, chunks)


# ---- CLI -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test end-to-end retrieval + generation")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    args = parser.parse_args()

    result = answer_question(args.query)

    print(f"\nQuestion: {args.query}\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Sources: {result['sources']}")
    print(f"Insufficient context: {result['insufficient']}")


if __name__ == "__main__":
    main()