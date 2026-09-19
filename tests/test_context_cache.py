import re

import gatekg.utils.context_cache as context_cache
from gatekg.utils.context_cache import (
    build_context_group_id,
    build_context_work_group_id,
    clear_context_group_manifest,
    has_context_group_manifest,
    mark_context_group_ready,
    normalize_context_chunks,
)


def test_normalize_context_chunks_drops_blank_entries():
    assert normalize_context_chunks(["alpha", "   ", "", "beta"]) == ["alpha", "beta"]


def test_group_id_is_stable_for_same_inputs():
    group_id_a = build_context_group_id(
        cache_model_key="ollama:qwen3.5:35b",
        context_chunks=["doc one", "doc two"],
    )
    group_id_b = build_context_group_id(
        cache_model_key="ollama:qwen3.5:35b",
        context_chunks=["doc one", "doc two"],
    )
    assert group_id_a == group_id_b
    assert ":" not in group_id_a
    assert re.fullmatch(r"[A-Za-z0-9_-]+", group_id_a)


def test_group_id_changes_when_model_key_changes():
    group_id_a = build_context_group_id(
        cache_model_key="ollama:qwen3.5:35b",
        context_chunks=["doc one", "doc two"],
    )
    group_id_b = build_context_group_id(
        cache_model_key="ollama:qwen3.5:70b",
        context_chunks=["doc one", "doc two"],
    )
    assert group_id_a != group_id_b


def test_work_group_id_is_safe_and_nonce_scoped():
    work_group_id = build_context_work_group_id(
        cache_model_key="ollama:qwen3.5:35b",
        context_chunks=["doc one", "doc two"],
        nonce="sessionabc123",
    )

    assert ":" not in work_group_id
    assert "sessionabc123" in work_group_id
    assert re.fullmatch(r"[A-Za-z0-9_-]+", work_group_id)


def test_context_group_manifest_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(context_cache, "CACHE_MANIFEST_DIR", tmp_path)
    group_id = "mcp-cache_testgroup"

    assert not has_context_group_manifest(group_id)

    manifest_path = mark_context_group_ready(
        group_id=group_id,
        cache_model_key="ollama:qwen3.5:35b",
        context_chunks=["doc one", "doc two"],
    )

    assert manifest_path == tmp_path / f"{group_id}.json"
    assert has_context_group_manifest(group_id)

    clear_context_group_manifest(group_id)
    assert not has_context_group_manifest(group_id)
