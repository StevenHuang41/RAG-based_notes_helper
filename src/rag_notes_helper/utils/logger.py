import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rag_notes_helper.core.config import get_settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    handler = RotatingFileHandler(
        get_settings().logs_dir / "rag.log",
        maxBytes=5_000_000,
        backupCount=2,
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

