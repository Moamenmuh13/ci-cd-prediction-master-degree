"""Shared utilities for the CI/CD failure prediction project.

Provides logging configuration and a few small path helpers. Reusable modules
should call ``get_logger(__name__)`` instead of using ``print``; standalone
scripts may print directly for the user-facing summary.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MODELS_DIR: Path = PROJECT_ROOT / "models"

RAW_DATASET_PATH: Path = RAW_DATA_DIR / "ci_cd_failures.csv"


_LOGGERS_CONFIGURED: set[str] = set()


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    The first call for a given ``name`` attaches a stdout ``StreamHandler`` and,
    if ``log_file`` is supplied, a ``FileHandler``. Subsequent calls return the
    same logger without re-adding handlers (avoiding duplicate log lines when a
    module is imported multiple times).
    """
    logger = logging.getLogger(name)
    if name in _LOGGERS_CONFIGURED:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS_CONFIGURED.add(name)
    return logger


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
