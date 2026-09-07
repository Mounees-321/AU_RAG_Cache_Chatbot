"""
chunker.py — Phase 2: Chunking Layer

Reads the raw scraped JSON files from data/raw/, splits each page's text
into overlapping chunks, and saves them as a single JSON list ready for
embedding in Phase 2b (vectorstore.py).

Usage:
    python chunker.py
Input:
    data/raw/*.json          (produced by ingest.py)
Output:
    data/chunks/chunks.json  (list of chunk records)
"""

import glob
import json
import logging
import os

try:
    # Newer LangChain versions (>=0.2) split this into its own package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Older LangChain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Config -------------------------------------------------------------

RAW_DIR = os.path.join("data", "raw")
OUTPUT_DIR = os.path.join("data", "chunks")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "chunks.json")

CHUNK_SIZE = 500       # target characters per chunk (roughly ~120-150 tokens)
CHUNK_OVERLAP = 75     # ~15% overlap, keeps context continuous across chunks
MIN_CHUNK_CHARS = 40   # drop tiny leftover fragments (e.g. stray nav text)


# ---- Core functions -------------------------------------------------------

def load_raw_pages(raw_dir: str) -> list[dict]:
    """Load every scraped page JSON file (skip the manifest)."""
    pages = []
    for filepath in glob.glob(os.path.join(raw_dir, "*.json")):
        if filepath.endswith("_manifest.json"):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            pages.append(json.load(f))
    return pages


def make_splitter() -> RecursiveCharacterTextSplitter:
    """
    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then sentences, then words — so chunks stay semantically coherent instead
    of cutting mid-sentence.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_page(page: dict, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """
    Split one page's text into chunk records, each carrying metadata back
    to its source. This metadata is what lets the chatbot cite sources later.
    """
    text = page.get("text", "")
    if not text:
        logger.warning(f"Empty text for {page.get('url')} — skipping")
        return []

    raw_chunks = splitter.split_text(text)

    chunk_records = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if len(chunk_text) < MIN_CHUNK_CHARS:
            continue  # skip fragments too small to be useful

        chunk_id = f"{page.get('url')}::chunk{i}"
        chunk_records.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "source_url": page.get("url"),
            "source_title": page.get("title"),
            "chunk_index": i,
        })

    return chunk_records


def main():
    logger.info(f"Loading raw pages from {RAW_DIR}")
    pages = load_raw_pages(RAW_DIR)

    if not pages:
        logger.error(f"No pages found in {RAW_DIR} — did you run ingest.py first?")
        return

    logger.info(f"Loaded {len(pages)} pages")

    splitter = make_splitter()
    all_chunks = []

    for page in pages:
        chunks = chunk_page(page, splitter)
        logger.info(f"{page.get('url')} -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.error("No chunks were produced — check your raw data and MIN_CHUNK_CHARS setting")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(all_chunks)} total chunks -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()