"""Logging setup for training scripts and library use."""

from __future__ import annotations

import logging
import sys
from typing import TextIO


def setup_logger(
    name: str,
    level: int = logging.INFO,
    stream: TextIO | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """Configure and return a named logger with a single stream handler.

    Idempotent for the same ``name``: if handlers already exist, returns the logger without
    adding duplicates (useful in notebooks / re-imports).

    Args:
        name: Logger name (often ``__name__`` of the entry script).
        level: Logging level.
        stream: Defaults to ``sys.stdout``.
        format_string: Log format; default includes level, name, and message.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(level)
    fmt = format_string or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
