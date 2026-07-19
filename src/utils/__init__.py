"""Common utility functions for seed settings and logging."""

import logging
import os
import random
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across random, numpy, and tensorflow.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Configure TensorFlow for deterministic execution if supported
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    logging.info(f"Random seed set to {seed} (Python, NumPy, TensorFlow)")


def setup_logger(log_dir: str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """Sets up a rotating file logger and UTF-8-safe console logger.

    Forces UTF-8 encoding on Windows CMD to prevent charmap codec errors when
    logging messages containing Unicode characters (dashes, arrows, etc).

    Args:
        log_dir (str): Directory where log files are stored.
        log_level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured root logger.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to prevent duplicate log entries
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ----------------------------------------------------------------
    # Console Handler — force UTF-8 to prevent charmap errors on Windows
    # ----------------------------------------------------------------
    try:
        # Python 3.7+: reconfigure stdout encoding in-place
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # Wrap emit() so any remaining UnicodeEncodeError is silently replaced
    _original_emit = console_handler.emit

    def _safe_emit(record):
        try:
            _original_emit(record)
        except UnicodeEncodeError:
            # Replace unencodable chars and retry
            try:
                record.msg = (
                    str(record.msg)
                    .encode("utf-8", errors="replace")
                    .decode(sys.stdout.encoding or "ascii", errors="replace")
                )
                _original_emit(record)
            except Exception:
                pass  # Never crash the training pipeline over a log entry

    console_handler.emit = _safe_emit
    root_logger.addHandler(console_handler)

    # ----------------------------------------------------------------
    # File Handlers
    # ----------------------------------------------------------------
    # 1. Training / General Info Log
    info_handler = RotatingFileHandler(
        log_path / "training.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    info_handler.setFormatter(formatter)
    info_handler.setLevel(logging.INFO)
    root_logger.addHandler(info_handler)

    # 2. Error Log
    error_handler = RotatingFileHandler(
        log_path / "error.log", maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8"
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_handler)

    logging.info(f"Logger initialized. Logs stored in: {log_path.resolve()}")
    return root_logger
