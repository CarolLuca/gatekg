import asyncio
import threading
import time

from gatekg.utils.llm_concurrency import LLMRequestLimiter


def test_async_limiter_serializes_concurrent_tasks():
    limiter = LLMRequestLimiter(1)
    state = {"current": 0, "max": 0}
    lock = threading.Lock()

    async def worker():
        async with limiter.async_slot():
            with lock:
                state["current"] += 1
                state["max"] = max(state["max"], state["current"])
            await asyncio.sleep(0.05)
            with lock:
                state["current"] -= 1

    async def run():
        await asyncio.gather(worker(), worker(), worker())

    asyncio.run(run())

    assert state["max"] == 1


def test_sync_limiter_serializes_concurrent_threads():
    limiter = LLMRequestLimiter(1)
    state = {"current": 0, "max": 0}
    lock = threading.Lock()

    def worker():
        with limiter.sync_slot():
            with lock:
                state["current"] += 1
                state["max"] = max(state["max"], state["current"])
            time.sleep(0.05)
            with lock:
                state["current"] -= 1

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state["max"] == 1
