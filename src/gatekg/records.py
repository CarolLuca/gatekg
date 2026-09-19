"""Portable experiment inputs stored alongside each run log."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gatekg.datasets import DatasetSpec
from gatekg.utils.config import load_config
from gatekg.utils.eval_runner import ExperimentEntry

SCHEMA_VERSION = "gatekg-experiment-v1"


def manifest_path(log_path: str | Path) -> Path:
    return Path(log_path).with_suffix(".manifest.json")


def write_run_manifest(
    log_path: str | Path,
    *,
    run_id: str,
    dataset: DatasetSpec,
    runner: str,
    judge: str,
    entries: list[ExperimentEntry],
) -> Path:
    """Snapshot inputs so training does not depend on the source dataset."""
    path = manifest_path(log_path)
    config = load_config()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "dataset": dataset.name,
        "dataset_path": str(dataset.path.resolve()),
        "runner": runner,
        "judge": judge,
        "dataset_options": {
            **asdict(dataset),
            "path": str(dataset.path.resolve()),
            "format": dataset.format.value,
        },
        "models": {
            "inference": config.active_model_key,
            "graph": config.kg_model_key,
            "embedding": f"{config.embedding_provider}:{config.emb_model_name}",
            "embedding_dim": config.embedding_dim,
            "judge": config.google_judge_model_name if judge == "google" else judge,
        },
        "entries": [asdict(entry) for entry in entries],
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def read_run_manifest(log_path: str | Path) -> dict[str, Any]:
    path = manifest_path(log_path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported experiment manifest: {path}")
    for key in ("dataset", "dataset_path", "runner", "judge"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"Manifest requires a nonempty {key}: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Manifest requires an entries list: {path}")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid entry in manifest: {path}")
        try:
            ExperimentEntry(**entry)
        except TypeError as exc:
            raise ValueError(f"Invalid entry in manifest: {path}") from exc
    return payload
