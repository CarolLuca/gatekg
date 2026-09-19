from __future__ import annotations

import contextlib
import logging
import re
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from gatekg.paths import runtime_paths
from gatekg.utils.config import load_config


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def _resolve_log_level(level_name: str, *, default: int) -> int:
    return getattr(logging, str(level_name).upper(), default)


class _TeeStream:
    def __init__(self, primary: TextIO, secondary: TextIO):
        self._primary = primary
        self._secondary = secondary

    @property
    def encoding(self) -> str | None:
        return getattr(self._primary, "encoding", None)

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._secondary.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        isatty = getattr(self._primary, "isatty", None)
        return bool(isatty()) if callable(isatty) else False

    def fileno(self) -> int:
        return self._primary.fileno()


def build_experiment_log_path(
    *,
    task_name: str,
    llm_name: str,
    model_key: str | None = None,
) -> Path:
    resolved_model_key = model_key or load_config().active_model_key
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    log_dir = (
        runtime_paths().logs
        / _slugify(task_name)
        / _slugify(llm_name)
        / _slugify(resolved_model_key)
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"run_{timestamp}.log"


@contextlib.contextmanager
def experiment_logging(
    *,
    task_name: str,
    runner_name: str,
    model_key: str | None = None,
    judge: str = "google",
) -> Iterator[Path]:
    """Capture a synchronous run and restore the caller's logging state."""
    config = load_config()
    log_path = build_experiment_log_path(
        task_name=task_name,
        llm_name=runner_name,
        model_key=model_key,
    )
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    levels = {
        "neo4j": config.logging.neo4j_level,
        "graphiti_core": config.logging.graphiti_level,
        "rlm": config.logging.rlm_level,
        "httpx": config.logging.http_level,
        "openai": config.logging.openai_level,
        "google": config.logging.google_level,
    }
    old_levels = {name: logging.getLogger(name).level for name in levels}
    with log_path.open("x", encoding="utf-8") as log_file:
        stdout = _TeeStream(sys.stdout, log_file)
        stderr = _TeeStream(sys.stderr, log_file)
        handler = logging.StreamHandler(stderr)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        root.handlers = [handler]
        root.setLevel(_resolve_log_level(config.logging.root_level, default=logging.INFO))
        for name, level in levels.items():
            logging.getLogger(name).setLevel(_resolve_log_level(level, default=logging.INFO))
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                print(f"Run log: {log_path}")
                print(f"Task: {task_name}")
                print(f"Runner: {runner_name}")
                print(f"Model: {model_key or config.active_model_key}")
                print(f"KG cache key: {config.kg_model_key}")
                print(
                    f"Judge model: {config.google_judge_model_name if judge == 'google' else judge}"
                )
                print("-----------------")
                yield log_path
        finally:
            handler.flush()
            handler.close()
            root.handlers = old_handlers
            root.setLevel(old_level)
            for name, old_logger_level in old_levels.items():
                logging.getLogger(name).setLevel(old_logger_level)
