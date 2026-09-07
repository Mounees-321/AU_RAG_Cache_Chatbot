"""
ingest.py — Phase 1: Ingestion Layer

Scrapes a list of Anna University department pages, strips boilerplate
(nav bars, footers), and saves clean text + metadata as JSON.

Usage:
    python ingest.py
Output:
    data/raw/<page_slug>.json   (one file per URL)
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Config -----------------------------------------------------------

URLS = [
    "https://departments.auegov.ac.in/it",
    "https://departments.auegov.ac.in/it/coursesoffered",
    "https://departments.auegov.ac.in/it/academicschedule",
    "https://departments.auegov.ac.in/it/teaching",
    "https://departments.auegov.ac.in/it/nonteaching",
    "https://departments.auegov.ac.in/it/deptfacilities",
    "https://departments.auegov.ac.in/it/studentsassociation",
    # --- Added: more sibling pages discovered from the site nav ---
    "https://departments.auegov.ac.in/it/pgstudents",
    "https://departments.auegov.ac.in/it/ugstudents",
    "https://departments.auegov.ac.in/it/phdawarded",
    "https://departments.auegov.ac.in/it/journals",
    "https://departments.auegov.ac.in/it/conferences",
    "https://departments.auegov.ac.in/it/projects",
    "https://departments.auegov.ac.in/it/announcements",
    # To add more yourself: visit https://departments.auegov.ac.in/it,
    # check the nav bar for new links, and append them here in the same format.
]

OUTPUT_DIR = os.path.join("data", "raw")
REQUEST_TIMEOUT = 15  # seconds
DELAY_BETWEEN_REQUESTS = 1.5  # seconds — be polite to the server
USER_AGENT = "Mozilla/5.0 (compatible; AU-RAG-Bot/1.0; educational project)"
    

# Tags that are almost always boilerplate, not real content
STRIP_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "iframe"]

# Text fragments that indicate boilerplate lines to drop (site-specific,
# tune this after inspecting your actual scraped output)
BOILERPLATE_SNIPPETS = [
    "All rights reserved",
    "Designed & Developed by",
    "Our Social Networks",
    "Useful Links",
    "Visit Count",
]


# ---- Core functions -----------------------------------------------------

def fetch_page(url: str) -> str:
    """Fetch raw HTML for a URL. Raises on network/HTTP errors."""
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def clean_html_to_text(html: str) -> tuple[str, str]:
    """
    Strip boilerplate tags and return (title, cleaned_text).
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Get visible text, collapse whitespace
    raw_text = soup.get_text(separator=" ", strip=True)
    lines = [line.strip() for line in raw_text.split("  ") if line.strip()]

    # Drop lines that are just boilerplate snippets
    filtered_lines = []
    for line in lines:
        if any(snippet.lower() in line.lower() for snippet in BOILERPLATE_SNIPPETS):
            continue
        filtered_lines.append(line)

    cleaned_text = " ".join(filtered_lines)
    # Collapse repeated whitespace
    cleaned_text = " ".join(cleaned_text.split())

    return title, cleaned_text


def url_to_slug(url: str) -> str:
    """Turn a URL into a filesystem-safe filename."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    return path if path else "index"


def save_page(url: str, title: str, text: str) -> str:
    """Save a single page's data as JSON. Returns the output filepath."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = url_to_slug(url)
    filepath = os.path.join(OUTPUT_DIR, f"{slug}.json")

    record = {
        "url": url,
        "title": title,
        "text": text,
        "char_count": len(text),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return filepath


def ingest_url(url: str) -> dict | None:
    """Fetch, clean, and save one URL. Returns the record dict, or None on failure."""
    try:
        logger.info(f"Fetching: {url}")
        html = fetch_page(url)
        title, text = clean_html_to_text(html)

        if len(text) < 50:
            logger.warning(f"Very little content extracted from {url} ({len(text)} chars) — check selectors")

        filepath = save_page(url, title, text)
        logger.info(f"Saved -> {filepath} ({len(text)} chars)")
        return {"url": url, "title": title, "char_count": len(text), "filepath": filepath}

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing {url}: {e}")
        return None


def main():
    logger.info(f"Starting ingestion of {len(URLS)} URLs")
    results = []

    for i, url in enumerate(URLS):
        result = ingest_url(url)
        if result:
            results.append(result)

        # Be polite — don't hammer the server. Skip delay after the last request.
        if i < len(URLS) - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    logger.info(f"Ingestion complete: {len(results)}/{len(URLS)} pages succeeded")

    # Write a manifest summarizing what was scraped — useful for the next phase
    manifest_path = os.path.join(OUTPUT_DIR, "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Manifest saved -> {manifest_path}")


if __name__ == "__main__":
    main()