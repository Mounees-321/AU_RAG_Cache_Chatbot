"""
logging_config.py — Phase 6: Centralized Logging

Instead of each module calling logging.basicConfig() separately (which
only the first call actually applies, silently ignoring the rest), this
module sets up logging ONCE, with both a console handler (for dev) and a
rotating file handler (for a persistent record of what happened -- useful
when debugging a query that failed hours ago, not just right now).

Usage (at the top of any script's main()):
    from logging_config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "rag_pipeline.log")

MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 MB per file
BACKUP_COUNT = 3                  # keep last 3 rotated log files

_configured = False  # guard against double-setup if multiple modules call this


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_LOG_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # keep more detail in the file than console

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Quiet down noisy third-party libraries so your own logs aren't buried
    for noisy_logger in ["httpx", "urllib3", "chromadb", "sentence_transformers"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _configured = True