from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from gatekg.paths import runtime_paths

CACHE_SCHEMA_VERSION = "mcp-v1"
GROUP_ID_PREFIX = "mcp-cache"
WORK_GROUP_ID_PREFIX = "mcp-work"
CACHE_MANIFEST_DIR = runtime_paths().cache / "context_graphs"


def normalize_context_chunks(context_chunks: list[str]) -> list[str]:
    normalized_chunks: list[str] = []
    for chunk in context_chunks:
        text = str(chunk)
        if not text.strip():
            continue
        normalized_chunks.append(text)
    return normalized_chunks


def build_context_cache_payload(
    *,
    cache_model_key: str,
    context_chunks: list[str],
) -> dict[str, object]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_model_key": cache_model_key,
        "context_chunks": normalize_context_chunks(context_chunks),
    }


def build_context_cache_digest(
    *,
    cache_model_key: str,
    context_chunks: list[str],
) -> str:
    payload = build_context_cache_payload(
        cache_model_key=cache_model_key,
        context_chunks=context_chunks,
    )
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def build_context_group_id(
    *,
    cache_model_key: str,
    context_chunks: list[str],
) -> str:
    digest = build_context_cache_digest(
        cache_model_key=cache_model_key,
        context_chunks=context_chunks,
    )
    # Graphiti group IDs only allow alphanumeric characters, dashes, and underscores.
    return f"{GROUP_ID_PREFIX}_{digest}"


def build_context_work_group_id(
    *,
    cache_model_key: str,
    context_chunks: list[str],
    nonce: str | None = None,
) -> str:
    digest = build_context_cache_digest(
        cache_model_key=cache_model_key,
        context_chunks=context_chunks,
    )
    work_nonce = nonce or uuid.uuid4().hex[:12]
    return f"{WORK_GROUP_ID_PREFIX}_{digest}_{work_nonce}"


def context_cache_manifest_path(group_id: str) -> Path:
    return CACHE_MANIFEST_DIR / f"{group_id}.json"


def has_context_group_manifest(group_id: str) -> bool:
    return context_cache_manifest_path(group_id).is_file()


def mark_context_group_ready(
    *,
    group_id: str,
    cache_model_key: str,
    context_chunks: list[str],
) -> Path:
    normalized_chunks = normalize_context_chunks(context_chunks)
    CACHE_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = context_cache_manifest_path(group_id)
    payload = build_context_cache_payload(
        cache_model_key=cache_model_key,
        context_chunks=normalized_chunks,
    )
    payload.update(
        {
            "group_id": group_id,
            "chunk_count": len(normalized_chunks),
            "built_at": datetime.now(UTC).isoformat(),
        }
    )
    temp_path = manifest_path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(manifest_path)
    return manifest_path


def clear_context_group_manifest(group_id: str) -> None:
    manifest_path = context_cache_manifest_path(group_id)
    if manifest_path.exists():
        manifest_path.unlink()
