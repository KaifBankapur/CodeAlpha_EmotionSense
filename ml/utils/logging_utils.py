"""Logging setup shared by the CLI, the trainer and the API."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str = "INFO", quiet_libraries: bool = True) -> None:
    """Configure root logging once, writing to stdout.

    Args:
        level: ``DEBUG``/``INFO``/``WARNING``/``ERROR``.
        quiet_libraries: raise the level of chatty third-party loggers (numba,
            librosa, matplotlib, urllib3) so the training log stays readable.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)

    if quiet_libraries:
        for name in (
            "numba",
            "librosa",
            "matplotlib",
            "matplotlib.pyplot",
            "urllib3",
            "matplotlib.font_manager",
            "PIL",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)

    # Keep warnings visible but non-fatal-looking.
    logging.captureWarnings(True)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def default_level() -> str:
    return os.environ.get("LOG_LEVEL", "INFO").upper()


__all__ = ["default_level", "get_logger", "setup_logging"]
