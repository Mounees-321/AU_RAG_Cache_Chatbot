"""
vectorstore.py — Phase 2: Vector Store Layer

Embeds chunks (from chunker.py output) and stores them in a persistent
ChromaDB collection. Also exposes a query() function used by later phases
(re-ranking, routing).

Usage (build the store from scratch):
    python vectorstore.py --build

Usage (quick manual test query):
    python vectorstore.py --query "What courses does the IT department offer?"

Input:
    data/chunks/chunks.json   (produced by chunker.py)
Output:
    data/chroma_db/           (persistent ChromaDB directory)
"""

import argparse
import json
import logging
import os

import chromadb
from chromadb.utils import embedding_functions

from exceptions import VectorStoreError
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ---- Config -------------------------------------------------------------

CHUNKS_FILE = os.path.join("data", "chunks", "chunks.json")
CHROMA_DIR = os.path.join("data", "chroma_db")
COLLECTION_NAME = "au_it_department"

# Free, local embedding model — no API key, no cost, runs on CPU.
# 'all-MiniLM-L6-v2' is small/fast; swap for 'bge-large-en-v1.5' if you
# want higher quality and don't mind slower embedding.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 20  # how many candidates to pull before re-ranking (Phase 3)


# ---- Core functions -------------------------------------------------------

def get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_or_create_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def load_chunks(chunks_file: str) -> list[dict]:
    if not os.path.exists(chunks_file):
        raise FileNotFoundError(
            f"{chunks_file} not found — run chunker.py first"
        )
    with open(chunks_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vectorstore():
    """Embed all chunks and add them to ChromaDB. Safe to re-run (upserts)."""
    logger.info("Loading chunks...")
    chunks = load_chunks(CHUNKS_FILE)
    logger.info(f"Loaded {len(chunks)} chunks")

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source_url": c["source_url"],
            "source_title": c["source_title"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    # Chroma's add() upserts by id, so re-running this is idempotent
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )
        logger.info(f"Embedded batch {i // batch_size + 1} ({min(i + batch_size, len(ids))}/{len(ids)})")

    logger.info(f"Vector store built. Collection count: {collection.count()}")


def query(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Query the vector store. Returns a list of candidate chunks with
    similarity scores, ready to be passed into the re-ranker (Phase 3).
    """
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)

        if collection.count() == 0:
            logger.warning("Vector store is empty — did you run vectorstore.py --build?")
            return []

        results = collection.query(
            query_texts=[question],
            n_results=min(top_k, collection.count()),  # avoid asking for more than exists
        )

        candidates = []
        for i in range(len(results["ids"][0])):
            candidates.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source_url": results["metadatas"][0][i].get("source_url"),
                "source_title": results["metadatas"][0][i].get("source_title"),
                "distance": results["distances"][0][i],  # lower = more similar (cosine distance)
            })

        return candidates

    except Exception as e:
        raise VectorStoreError("query", str(e))


# ---- CLI -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build or query the RAG vector store")
    parser.add_argument("--build", action="store_true", help="Embed chunks and build the vector store")
    parser.add_argument("--query", type=str, help="Run a test query against the vector store")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to show for --query")
    args = parser.parse_args()

    if args.build:
        build_vectorstore()
    elif args.query:
        results = query(args.query, top_k=args.top_k)
        print(f"\nTop {len(results)} results for: {args.query!r}\n")
        for i, r in enumerate(results, 1):
            print(f"--- Result {i} (distance={r['distance']:.4f}) ---")
            print(f"Source: {r['source_title']} ({r['source_url']})")
            print(f"Text: {r['text'][:200]}...")
            print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()