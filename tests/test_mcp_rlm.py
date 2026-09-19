import asyncio

from gatekg.api.mcp_rlm import (
    BaseRLM,
    CustomBaseRLM,
    ReadWriteRLM,
    _KGEnabledMCPRLM,
    _KGMethodMixin,
)


class _DummyResponse:
    def __init__(self, response: str):
        self.response = response


class _DummyClient:
    def completion(self, text: str):
        return _DummyResponse(text)


def test_base_and_custom_base_do_not_expose_kg(monkeypatch):
    import gatekg.api.mcp_rlm as mcp_rlm

    monkeypatch.setattr(mcp_rlm, "build_rlm_client", lambda **_: _DummyClient())

    base = BaseRLM()
    custom = CustomBaseRLM()

    assert not hasattr(base, "kg")
    assert not hasattr(base, "search_nodes")
    assert not hasattr(custom, "kg")
    assert hasattr(custom, "context")
    assert hasattr(custom, "set_context")


def test_read_write_rlm_exposes_context_and_kg_tools(monkeypatch):
    import gatekg.api.mcp_rlm as mcp_rlm

    captured_kwargs = {}

    class FakeKG:
        active_group_id = None
        active_group_ids = None

    def fake_build_rlm_client(**kwargs):
        captured_kwargs.update(kwargs)
        return _DummyClient()

    monkeypatch.setattr(mcp_rlm, "build_rlm_client", fake_build_rlm_client)
    monkeypatch.setattr(mcp_rlm, "MCP_KG", FakeKG)
    monkeypatch.setattr(mcp_rlm, "_tool_description", lambda name: name)

    rlm = ReadWriteRLM()

    assert hasattr(rlm, "kg")
    assert hasattr(rlm, "context")
    assert hasattr(rlm, "set_context")
    assert hasattr(rlm, "search_nodes")
    assert hasattr(rlm, "search_memory_facts")
    assert hasattr(rlm, "add_memory")
    assert "get_supporting_facts" in captured_kwargs["custom_tools"]
    assert "add_memory" in captured_kwargs["custom_tools"]
    assert "search_nodes" in captured_kwargs["custom_tools"]


def test_close_kg_is_non_destructive():
    calls: list[str] = []

    class FakeKG:
        async def close(self):
            calls.append("close")

        async def clear_graph(self, group_ids=None):
            calls.append("clear_graph")

    class DummyRLM(_KGMethodMixin):
        def __init__(self):
            self.kg = FakeKG()

    asyncio.run(DummyRLM().close_kg())
    assert calls == ["close"]


def test_prepare_context_builds_on_cache_miss(monkeypatch):
    import gatekg.api.mcp_rlm as mcp_rlm

    class FakeKG:
        def __init__(self):
            self.active_group_id = None
            self.active_group_ids = None
            self.has_group = False
            self.init_calls = []

        def set_active_group(self, group_id):
            self.active_group_id = group_id

        def set_active_groups(self, group_ids):
            self.active_group_ids = list(group_ids) if group_ids is not None else None

        async def has_group_data(self, group_id=None):
            return self.has_group

        async def init_knowledge(self, supporting_facts, group_id=None):
            self.init_calls.append((list(supporting_facts), group_id))

    class DummyRLM:
        prepare_context = _KGEnabledMCPRLM.prepare_context

        def __init__(self):
            self.context = []
            self.kg = FakeKG()
            self.init_count = 0
            self.cleared = []
            self.writable_tools = False
            self.base_group_id = None
            self.write_group_id = None
            self._ephemeral_group_ids = set()

        def set_context(self, texts):
            self.context = list(texts)

        async def init_kg(self):
            self.init_count += 1

        async def clear_cached_group(self, group_id=None):
            self.cleared.append(group_id)

    ready_calls = []
    monkeypatch.setattr(mcp_rlm, "has_context_group_manifest", lambda group_id: False)
    monkeypatch.setattr(
        mcp_rlm,
        "mark_context_group_ready",
        lambda **kwargs: ready_calls.append(kwargs),
    )

    rlm = DummyRLM()
    group_id = asyncio.run(
        rlm.prepare_context(
            ["alpha", "", "beta"],
            cache_model_key="ollama:qwen3.5:35b",
        )
    )

    assert rlm.context == ["alpha", "beta"]
    assert rlm.kg.active_group_id == group_id
    assert rlm.kg.active_group_ids == [group_id]
    assert rlm.kg.init_calls == [(["alpha", "beta"], group_id)]
    assert rlm.init_count == 1
    assert ready_calls == [
        {
            "group_id": group_id,
            "cache_model_key": "ollama:qwen3.5:35b",
            "context_chunks": ["alpha", "beta"],
        }
    ]


def test_prepare_context_defaults_to_rlm_kg_model_key(monkeypatch):
    import gatekg.api.mcp_rlm as mcp_rlm

    class FakeKG:
        def __init__(self):
            self.active_group_id = None
            self.active_group_ids = None
            self.init_calls = []

        def set_active_group(self, group_id):
            self.active_group_id = group_id

        def set_active_groups(self, group_ids):
            self.active_group_ids = list(group_ids) if group_ids is not None else None

        async def has_group_data(self, group_id=None):
            return False

        async def init_knowledge(self, supporting_facts, group_id=None):
            self.init_calls.append((list(supporting_facts), group_id))

    class DummyRLM:
        prepare_context = _KGEnabledMCPRLM.prepare_context

        def __init__(self):
            self.context = []
            self.kg = FakeKG()
            self.init_count = 0
            self.cleared = []
            self.writable_tools = False
            self.base_group_id = None
            self.write_group_id = None
            self._ephemeral_group_ids = set()
            self.kg_model_key = "google:gemini-3-flash-preview"

        def set_context(self, texts):
            self.context = list(texts)

        async def init_kg(self):
            self.init_count += 1

        async def clear_cached_group(self, group_id=None):
            self.cleared.append(group_id)

    ready_calls = []
    monkeypatch.setattr(mcp_rlm, "has_context_group_manifest", lambda group_id: False)
    monkeypatch.setattr(
        mcp_rlm,
        "mark_context_group_ready",
        lambda **kwargs: ready_calls.append(kwargs),
    )

    rlm = DummyRLM()
    asyncio.run(rlm.prepare_context(["alpha", "beta"]))

    assert ready_calls == [
        {
            "group_id": rlm.base_group_id,
            "cache_model_key": "google:gemini-3-flash-preview",
            "context_chunks": ["alpha", "beta"],
        }
    ]


def test_prepare_context_skips_build_on_cache_hit(monkeypatch):
    import gatekg.api.mcp_rlm as mcp_rlm

    class FakeKG:
        def __init__(self):
            self.active_group_id = None
            self.active_group_ids = None

        def set_active_group(self, group_id):
            self.active_group_id = group_id

        def set_active_groups(self, group_ids):
            self.active_group_ids = list(group_ids) if group_ids is not None else None

        async def has_group_data(self, group_id=None):
            return True

        async def init_knowledge(self, supporting_facts, group_id=None):
            raise AssertionError("init_knowledge should not be called on cache hit")

    class DummyRLM:
        prepare_context = _KGEnabledMCPRLM.prepare_context

        def __init__(self):
            self.context = []
            self.kg = FakeKG()
            self.writable_tools = False
            self.base_group_id = None
            self.write_group_id = None
            self._ephemeral_group_ids = set()

        def set_context(self, texts):
            self.context = list(texts)

        async def init_kg(self):
            return None

        async def clear_cached_group(self, group_id=None):
            raise AssertionError("clear_cached_group should not be called without rebuild")

    monkeypatch.setattr(mcp_rlm, "has_context_group_manifest", lambda group_id: True)
    monkeypatch.setattr(
        mcp_rlm,
        "mark_context_group_ready",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("mark_context_group_ready should not be called on cache hit")
        ),
    )

    rlm = DummyRLM()
    asyncio.run(
        rlm.prepare_context(
            ["alpha", "beta"],
            cache_model_key="ollama:qwen3.5:35b",
        )
    )
    assert rlm.context == ["alpha", "beta"]
    assert rlm.kg.active_group_ids == [rlm.kg.active_group_id]


def test_prepare_context_clears_stale_partial_cache_without_manifest(monkeypatch):
    import gatekg.api.mcp_rlm as mcp_rlm

    class FakeKG:
        def __init__(self):
            self.active_group_id = None
            self.active_group_ids = None
            self.init_calls = []

        def set_active_group(self, group_id):
            self.active_group_id = group_id

        def set_active_groups(self, group_ids):
            self.active_group_ids = list(group_ids) if group_ids is not None else None

        async def has_group_data(self, group_id=None):
            return True

        async def init_knowledge(self, supporting_facts, group_id=None):
            self.init_calls.append((list(supporting_facts), group_id))

    class DummyRLM:
        prepare_context = _KGEnabledMCPRLM.prepare_context

        def __init__(self):
            self.context = []
            self.kg = FakeKG()
            self.writable_tools = False
            self.base_group_id = None
            self.write_group_id = None
            self._ephemeral_group_ids = set()
            self.cleared = []

        def set_context(self, texts):
            self.context = list(texts)

        async def init_kg(self):
            return None

        async def clear_cached_group(self, group_id=None):
            self.cleared.append(group_id)

    ready_calls = []
    monkeypatch.setattr(mcp_rlm, "has_context_group_manifest", lambda group_id: False)
    monkeypatch.setattr(
        mcp_rlm,
        "mark_context_group_ready",
        lambda **kwargs: ready_calls.append(kwargs),
    )

    rlm = DummyRLM()
    group_id = asyncio.run(
        rlm.prepare_context(
            ["alpha", "beta"],
            cache_model_key="ollama:qwen3.5:35b",
        )
    )

    assert rlm.cleared == [group_id]
    assert rlm.kg.init_calls == [(["alpha", "beta"], group_id)]
    assert ready_calls == [
        {
            "group_id": group_id,
            "cache_model_key": "ollama:qwen3.5:35b",
            "context_chunks": ["alpha", "beta"],
        }
    ]


def test_prepare_context_uses_separate_work_group_for_writable_mode(monkeypatch):
    import gatekg.api.mcp_rlm as mcp_rlm

    class FakeKG:
        def __init__(self):
            self.active_group_id = None
            self.active_group_ids = None
            self.init_calls = []

        def set_active_group(self, group_id):
            self.active_group_id = group_id

        def set_active_groups(self, group_ids):
            self.active_group_ids = list(group_ids) if group_ids is not None else None

        async def has_group_data(self, group_id=None):
            return True

        async def init_knowledge(self, supporting_facts, group_id=None):
            self.init_calls.append((list(supporting_facts), group_id))

        async def clear_graph(self, group_ids=None):
            return None

        async def close(self):
            return None

    monkeypatch.setattr(mcp_rlm, "build_context_work_group_id", lambda **_: "mcp-work_session123")
    monkeypatch.setattr(mcp_rlm, "has_context_group_manifest", lambda group_id: True)
    monkeypatch.setattr(
        mcp_rlm,
        "mark_context_group_ready",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("mark_context_group_ready should not be called on cache hit")
        ),
    )

    class DummyRLM:
        prepare_context = _KGEnabledMCPRLM.prepare_context

        def __init__(self):
            self.context = []
            self.kg = FakeKG()
            self.writable_tools = True
            self.base_group_id = None
            self.write_group_id = None
            self._ephemeral_group_ids = set()

        def set_context(self, texts):
            self.context = list(texts)

        async def init_kg(self):
            return None

        async def clear_cached_group(self, group_id=None):
            raise AssertionError("clear_cached_group should not be called without rebuild")

    rlm = DummyRLM()
    base_group_id = asyncio.run(
        rlm.prepare_context(
            ["alpha", "beta"],
            cache_model_key="ollama:qwen3.5:35b",
        )
    )

    assert base_group_id.startswith("mcp-cache_")
    assert rlm.base_group_id == base_group_id
    assert rlm.write_group_id == "mcp-work_session123"
    assert rlm.kg.active_group_id == "mcp-work_session123"
    assert rlm.kg.active_group_ids == ["mcp-work_session123", base_group_id]


def test_prepare_context_uses_runtime_context_for_supporting_facts_and_kg_context_for_cache(
    monkeypatch,
):
    import gatekg.api.mcp_rlm as mcp_rlm

    class FakeKG:
        def __init__(self):
            self.active_group_id = None
            self.active_group_ids = None
            self.init_calls = []

        def set_active_group(self, group_id):
            self.active_group_id = group_id

        def set_active_groups(self, group_ids):
            self.active_group_ids = list(group_ids) if group_ids is not None else None

        async def has_group_data(self, group_id=None):
            return False

        async def init_knowledge(self, supporting_facts, group_id=None):
            self.init_calls.append((list(supporting_facts), group_id))

    class DummyRLM:
        prepare_context = _KGEnabledMCPRLM.prepare_context

        def __init__(self):
            self.context = []
            self.kg = FakeKG()
            self.init_count = 0
            self.writable_tools = False
            self.base_group_id = None
            self.write_group_id = None
            self._ephemeral_group_ids = set()

        def set_context(self, texts):
            self.context = list(texts)

        async def init_kg(self):
            self.init_count += 1

        async def clear_cached_group(self, group_id=None):
            raise AssertionError("clear_cached_group should not be called on cache miss")

    ready_calls = []
    monkeypatch.setattr(mcp_rlm, "has_context_group_manifest", lambda group_id: False)
    monkeypatch.setattr(
        mcp_rlm,
        "mark_context_group_ready",
        lambda **kwargs: ready_calls.append(kwargs),
    )

    rlm = DummyRLM()
    group_id = asyncio.run(
        rlm.prepare_context(
            ["extended-a", "", "extended-b"],
            kg_context_chunks=["original-a", "", "original-b"],
            cache_model_key="ollama:qwen3.5:35b",
        )
    )

    assert rlm.context == ["extended-a", "extended-b"]
    assert rlm.kg.init_calls == [(["original-a", "original-b"], group_id)]
    assert ready_calls == [
        {
            "group_id": group_id,
            "cache_model_key": "ollama:qwen3.5:35b",
            "context_chunks": ["original-a", "original-b"],
        }
    ]


def test_prepare_context_records_baseline_health_snapshot(monkeypatch):
    import gatekg.api.mcp_rlm as mcp_rlm

    class FakeKG:
        def __init__(self):
            self.active_group_id = None
            self.active_group_ids = None

        def set_active_group(self, group_id):
            self.active_group_id = group_id

        def set_active_groups(self, group_ids):
            self.active_group_ids = list(group_ids) if group_ids is not None else None

        async def has_group_data(self, group_id=None):
            return True

    class DummyRLM:
        prepare_context = _KGEnabledMCPRLM.prepare_context

        def __init__(self):
            self.context = []
            self.kg = FakeKG()
            self.writable_tools = False
            self.base_group_id = None
            self.write_group_id = None
            self._ephemeral_group_ids = set()

        def set_context(self, texts):
            self.context = list(texts)

        async def init_kg(self):
            return None

        async def clear_cached_group(self, group_id=None):
            raise AssertionError("clear_cached_group should not be called on cache hit")

    baseline_calls = []

    async def fake_ensure_baseline_health_snapshot(**kwargs):
        baseline_calls.append(kwargs)
        return {"snapshot_id": "baseline:test"}

    monkeypatch.setattr(mcp_rlm, "has_context_group_manifest", lambda group_id: True)
    monkeypatch.setattr(
        mcp_rlm, "ensure_baseline_health_snapshot", fake_ensure_baseline_health_snapshot
    )

    rlm = DummyRLM()
    asyncio.run(
        rlm.prepare_context(
            ["alpha", "", "beta"],
            cache_model_key="ollama:qwen3.5:35b",
        )
    )

    assert baseline_calls == [
        {
            "kg": rlm.kg,
            "base_group_id": rlm.base_group_id,
            "cache_model_key": "ollama:qwen3.5:35b",
            "context_chunk_count": 2,
            "force": False,
            "metadata": {"rlm_class": "DummyRLM"},
        }
    ]


def test_after_predict_records_run_health_snapshot(monkeypatch):
    import gatekg.api.mcp_rlm as mcp_rlm

    run_calls = []

    async def fake_record_run_health_snapshot(**kwargs):
        run_calls.append(kwargs)
        return {"snapshot_id": "run:test"}

    monkeypatch.setattr(mcp_rlm, "record_run_health_snapshot", fake_record_run_health_snapshot)

    class DummyRLM:
        after_predict = _KGEnabledMCPRLM.after_predict

        def __init__(self):
            self.kg = object()
            self.writable_tools = True
            self.base_group_id = "mcp-cache_base"
            self.write_group_id = "mcp-work_run"
            self.model_key = "google:gemini"
            self.kg_model_key = "google:gemini-kg"

    rlm = DummyRLM()
    asyncio.run(
        rlm.after_predict(
            prompt="prompt text",
            question="question text",
            prediction="answer text",
        )
    )

    assert run_calls == [
        {
            "kg": rlm.kg,
            "base_group_id": "mcp-cache_base",
            "write_group_id": "mcp-work_run",
            "model_key": "google:gemini",
            "kg_model_key": "google:gemini-kg",
            "question": "question text",
            "prompt": "prompt text",
            "prediction": "answer text",
            "metadata": {"rlm_class": "DummyRLM"},
        }
    ]


def test_writable_close_cleans_ephemeral_groups():
    calls: list[tuple[str, list[str] | None]] = []

    class FakeKG:
        async def clear_graph(self, group_ids=None):
            calls.append(("clear_graph", list(group_ids) if group_ids is not None else None))

        async def close(self):
            calls.append(("close", None))

    class DummyRLM(_KGEnabledMCPRLM):
        def __init__(self):
            self.kg = FakeKG()
            self.writable_tools = True
            self.base_group_id = None
            self.write_group_id = None
            self._ephemeral_group_ids = {"mcp-work_a", "mcp-work_b"}

    asyncio.run(DummyRLM().close_kg())
    assert calls[-1] == ("close", None)
    assert ("clear_graph", ["mcp-work_a"]) in calls
    assert ("clear_graph", ["mcp-work_b"]) in calls


def test_rlm_exception_propagates():
    from unittest.mock import Mock

    import pytest

    from gatekg.api.mcp_rlm import BaseRLM

    model = BaseRLM.__new__(BaseRLM)
    model.client = Mock()
    model.client.completion.side_effect = ConnectionError("inference offline")
    with pytest.raises(ConnectionError, match="inference offline"):
        model.predict("context", "q")
