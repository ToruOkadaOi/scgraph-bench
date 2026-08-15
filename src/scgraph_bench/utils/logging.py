"""Structured logging utility for scgraph-bench."""

from __future__ import annotations

import logging
import sys
from typing import Any

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str | int = "INFO",
    log_file: str | None = None,
    rich_tracebacks: bool = True,
) -> logging.Logger:
    """Initialize structured application logging.

    Args:
        level: Logging level (e.g. "DEBUG", "INFO", "WARNING").
        log_file: Optional file destination for logs.
        rich_tracebacks: Whether to format tracebacks using rich.

    Returns:
        Root scgraph-bench logger.
    """
    logger = logging.getLogger("scgraph_bench")
    logger.setLevel(level)

    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        console = Console(file=sys.stderr)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=rich_tracebacks,
            markup=True,
        )
        formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="[%Y-%m-%d %H:%M:%S]",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a child logger under the scgraph_bench namespace."""
    if name:
        return logging.getLogger(f"scgraph_bench.{name}")
    return logging.getLogger("scgraph_bench")


class BenchmarkLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter to attach run/split context automatically."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        extra = self.extra or {}
        context_str = " ".join(f"[{k}={v}]" for k, v in extra.items())
        if context_str:
            return f"{context_str} {msg}", kwargs
        return msg, kwargs
