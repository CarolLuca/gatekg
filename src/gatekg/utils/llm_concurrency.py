from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

from gatekg.utils.config import Config, load_config

config: Config = load_config()


@dataclass(frozen=True)
class LLMConcurrencyConfig:
    max_coroutines: int | None


class LLMRequestLimiter:
    def __init__(self, max_coroutines: int | None):
        self.max_coroutines = max_coroutines
        self._semaphore = (
            threading.BoundedSemaphore(max_coroutines)
            if max_coroutines is not None and max_coroutines > 0
            else None
        )

    @contextmanager
    def sync_slot(self):
        if self._semaphore is None:
            yield
            return

        self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()

    @asynccontextmanager
    async def async_slot(self):
        if self._semaphore is None:
            yield
            return

        while not self._semaphore.acquire(blocking=False):
            await asyncio.sleep(0.001)
        try:
            yield
        finally:
            self._semaphore.release()


LLM_CONCURRENCY = LLMConcurrencyConfig(max_coroutines=config.graphiti.max_coroutines)
LLM_REQUEST_LIMITER = LLMRequestLimiter(LLM_CONCURRENCY.max_coroutines)
