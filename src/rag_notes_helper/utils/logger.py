import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from pydantic import ValidationError

from rag_notes_helper.core.config import get_settings


def _safe_logs_dir() -> Path:
    try :
        return get_settings().logs_dir
    except ValidationError:
        fallback_log_path = Path.cwd() / "logs"
        fallback_log_path.mkdir(parents=True, exist_ok=True)
        return fallback_log_path

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    log_path = _safe_logs_dir() / "rag.log"

    handler = RotatingFileHandler(
        log_path,
        maxBytes=5_000_000,
        backupCount=2,
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

