"""Bounded retries for synchronous provider calls."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


def call_with_retries(operation: Callable[[], T], *, attempts: int, delay_seconds: float) -> T:
    """Retry failures without turning exhausted requests into model answers."""
    if attempts < 1 or delay_seconds < 0:
        raise ValueError("Retry attempts must be positive and delay must be nonnegative")
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception:
            if attempt == attempts:
                raise
            logger.warning("Provider call failed; retrying (%s/%s)", attempt, attempts)
            time.sleep(min(60.0, delay_seconds * attempt))
    raise AssertionError("Retry loop completed without a result")
