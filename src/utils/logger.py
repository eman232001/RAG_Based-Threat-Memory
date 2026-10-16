"""
src/utils/logger.py
Structured logging — writes to both console and logs/run.log.
Usage: from src.utils.logger import get_logger
       log = get_logger(__name__)
       log.info("message")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Return a logger that writes to stdout and optionally to a file.

    Args:
        name: Logger name (use __name__ in calling module).
        log_file: Path to log file. If None, uses logs/run.log relative to CWD.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (reconfigure stdout for utf-8 if available)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file is None:
        log_file = Path("logs") / "run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    _LOGGERS[name] = logger
    return logger
