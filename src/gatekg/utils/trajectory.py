"""Bounded, owner-readable evidence for training on graph tool calls."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, Any
from uuid import UUID

from pydantic import BaseModel

SCHEMA = "gatekg-tool-trace-v1"
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "cookie",
    "set_cookie",
    "credentials",
}


def trace_path(log_path: Path) -> Path:
    return log_path.with_suffix(".trajectory.jsonl")


@contextmanager
def private_output(path: Path) -> Iterator[IO[str]]:
    """Atomically replace a derived file with an owner-only copy."""
    with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temp = Path(handle.name)
        try:
            yield handle.file
            handle.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def capture_value(value: Any) -> Any:
    # Bound both text and object traversal, including deeply nested results.
    remaining_chars, remaining_items = 8192, 128

    def visit(item: Any, depth: int) -> Any:
        nonlocal remaining_chars, remaining_items
        remaining_items -= 1
        if depth > 8 or remaining_items < 0:
            return {"omitted": "capture limit"}
        if isinstance(item, BaseModel):
            item = {key: getattr(item, key) for key in type(item).model_fields}
        if isinstance(item, (date, UUID)):
            item = str(item)
        if isinstance(item, str):
            size = min(len(item), remaining_chars, 4096)
            remaining_chars -= size
            return (
                item
                if size == len(item)
                else {"text": item[:size], "omitted_chars": len(item) - size}
            )
        if item is None or isinstance(item, (int, bool)):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else str(item)
        if isinstance(item, dict):
            result = {}
            for key, val in islice(item.items(), 32):
                if remaining_items <= 0:
                    break
                name = str(key)[:128]
                normalized = name.casefold().replace("-", "_")
                if normalized in SENSITIVE_KEYS or normalized.endswith(
                    ("_api_key", "_password", "_token", "_secret")
                ):
                    result[name] = "[REDACTED]"
                else:
                    result[name] = visit(val, depth + 1)
            if len(result) < len(item):
                return {"items": result, "omitted_items": len(item) - len(result)}
            return result
        if isinstance(item, (list, tuple)):
            items = []
            for val in item[:32]:
                if remaining_items <= 0:
                    break
                items.append(visit(val, depth + 1))
            return (
                items
                if len(items) == len(item)
                else {"items": items, "omitted_items": len(item) - len(items)}
            )
        return {"omitted_type": type(item).__name__}

    return visit(value, 0)


@contextmanager
def capture_entry(model: Any, path: Path | None, entry_id: str) -> Iterator[None]:
    if path is None:
        yield
        return
    attempts: list[dict[str, Any]] = []
    model._trajectory_attempts = attempts
    try:
        yield
    finally:
        model._trajectory_attempts = None
        with open(
            path, "a", encoding="utf-8", opener=lambda name, flags: os.open(name, flags, 0o600)
        ) as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "entry_id": entry_id,
                        "selected_variant": getattr(model, "selected_variant", None),
                        "attempts": attempts,
                    },
                    allow_nan=False,
                )
                + "\n"
            )


def read_traces(path: Path) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            entry_id = record["entry_id"]
            if record["schema"] != SCHEMA or entry_id in traces:
                raise ValueError(f"Invalid or duplicate trajectory for {entry_id!r} in {path}")
            traces[entry_id] = record
    return traces
