"""Minimal, dependency-free logging setup.

Call `setup_logging()` once on startup. Anywhere else, just do:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("hello")
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger to print clean, single-line logs to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if this is called more than once (e.g. in tests).
    root.handlers.clear()
    root.addHandler(handler)
