from __future__ import annotations

import asyncio
import json
import logging
from stat import S_IMODE
from types import SimpleNamespace

import pytest

from gatekg.datasets import DatasetSpec
from gatekg.records import write_run_manifest
from gatekg.training.dataset import (
    build_gate_examples,
    build_trajectory_gate_examples,
    parse_run_log,
)
from gatekg.training.label import _cache_key, _prompt_for_call
from gatekg.utils.eval_runner import ExperimentEntry
from gatekg.utils.prompts import READ_WRITE_RLM_SYSTEM_PROMPT
from gatekg.utils.trajectory import capture_entry, read_traces, trace_path


class _DummyResponse:
    def __init__(self, response: str):
        self.response = response


class _DummyClient:
    def __init__(self):
        self.custom_tools = {}
        self.custom_sub_tools = {}
        self.system_prompt = ""

    def completion(self, prompt: str, root_prompt: str | None = None):
        return _DummyResponse(prompt)


class _FakeKG:
    def __init__(self):
        self.active_group_id = None
        self.active_group_ids = None
        self.init_calls = []
        self.search_calls = []

    async def initialize(self):
        return None

    async def close(self):
        return None

    async def has_group_data(self, group_id=None):
        return False

    async def init_knowledge(self, supporting_facts, group_id=None):
        self.init_calls.append((list(supporting_facts), group_id))

    async def clear_graph(self, group_ids=None):
        return None

    def set_active_group(self, group_id):
        self.active_group_id = group_id

    def set_active_groups(self, group_ids):
        self.active_group_ids = list(group_ids)

    async def search_nodes(self, *args, **kwargs):
        self.search_calls.append(("search_nodes", args, kwargs))
        return [{"name": "node"}]

    async def add_memory(self, *args, **kwargs):
        self.search_calls.append(("add_memory", args, kwargs))
        return {"ok": True}


@pytest.fixture
def graph_policy(monkeypatch):
    import gatekg.api.gatekg as gatekg
    import gatekg.api.mcp_rlm as mcp_rlm

    recorded_run_health = []

    async def fake_ensure_baseline_health_snapshot(**kwargs):
        return {
            "snapshot_id": f"baseline:{kwargs['base_group_id']}",
            "average_health_score": 0.8,
            "metric_scores": {"coverage": 0.7},
        }

    async def fake_record_run_health_snapshot(**kwargs):
        recorded_run_health.append(kwargs)
        return {"snapshot_id": "run:test"}

    monkeypatch.setattr(
        gatekg,
        "load_inference_bundle",
        lambda *args, **kwargs: SimpleNamespace(feature_config=SimpleNamespace(gate_threshold=0.5)),
    )
    monkeypatch.setattr(mcp_rlm, "build_rlm_client", lambda **kwargs: _DummyClient())
    monkeypatch.setattr(mcp_rlm, "MCP_KG", _FakeKG)
    monkeypatch.setattr(gatekg, "_tool_description", lambda name: name)
    monkeypatch.setattr(gatekg, "has_context_group_manifest", lambda group_id: False)
    monkeypatch.setattr(gatekg, "mark_context_group_ready", lambda **kwargs: None)
    monkeypatch.setattr(gatekg, "clear_context_group_manifest", lambda group_id: None)
    monkeypatch.setattr(
        gatekg, "ensure_baseline_health_snapshot", fake_ensure_baseline_health_snapshot
    )
    monkeypatch.setattr(gatekg, "record_run_health_snapshot", fake_record_run_health_snapshot)
    monkeypatch.setattr(
        gatekg,
        "KGHealthSnapshotStore",
        lambda: SimpleNamespace(get_baseline_snapshot=lambda group_id: None),
    )

    rlm = gatekg.GateKG(dataset_variant="base", verbose=False)
    monkeypatch.setattr(
        rlm,
        "_predict_selector",
        lambda question: {
            "read_only_rlm": 0.1,
            "read_write_rlm": 0.9,
            "aggressive_write_rlm": 0.2,
        },
    )
    monkeypatch.setattr(
        rlm,
        "_predict_gate",
        lambda **kwargs: 0.9 if kwargs["tool_name"] == "search_nodes" else 0.1,
    )

    asyncio.run(rlm.prepare_context(["alpha", "beta"], question="Which node?"))

    return rlm, recorded_run_health


def test_gatekg_prepare_context_selects_variant_and_gates_tools(graph_policy):
    rlm, recorded_run_health = graph_policy
    assert rlm.selected_variant == "read_write_rlm"
    assert rlm.writable_tools is True
    assert "add_memory" in rlm.client.custom_tools
    assert rlm.client.system_prompt == READ_WRITE_RLM_SYSTEM_PROMPT

    allowed_result = rlm.client.custom_tools["search_nodes"]["tool"]("alpha", max_nodes=3)
    blocked_result = rlm.client.custom_tools["add_memory"]["tool"](
        name="memory",
        episode_body="body",
        source_description="source",
    )

    assert allowed_result == [{"name": "node"}]
    assert blocked_result["blocked"] is True
    assert blocked_result["tool"] == "add_memory"

    asyncio.run(
        rlm.after_predict(
            prompt="prompt",
            question="Which node?",
            prediction="alpha",
        )
    )
    assert recorded_run_health


@pytest.mark.parametrize("structured", [True, False], ids=["private-trace", "legacy-log"])
def test_blocked_proposal_survives_dataset_and_gate_training(
    graph_policy, caplog, tmp_path, structured
):
    rlm, _ = graph_policy
    caplog.set_level(logging.INFO)
    log = tmp_path / "run.log"
    selection = {"selected_variant": rlm.selected_variant}
    entry = ExperimentEntry("one", "Which node?", "London", ["a"])
    write_run_manifest(
        log,
        run_id="sample",
        dataset=DatasetSpec("sample", tmp_path / "qa"),
        runner="gatekg",
        judge="exact",
        entries=[entry],
    )
    trace = trace_path(log)
    with capture_entry(rlm, trace if structured else None, entry.entry_id):
        tools = rlm.client.custom_tools
        tools["add_memory"]["tool"](name="fact", episode_body="sensitive memory")
        tools["search_nodes"]["tool"]("London", max_nodes=3)
        tools["search_nodes"]["tool"](query="Paris")
    log.write_text(
        "\n".join(
            [
                "Model: example",
                "Testing entry 1/1",
                "GateKG selection event: " + json.dumps(selection),
                *caplog.messages,
                "-----------------",
                "Which node?",
                "London",
                "-----------------",
                "London",
                "-----------------",
                "judge score: 1",
                "judge explanation: matched",
                "-----------------",
            ]
        )
    )
    assert [name for name, _, _ in rlm.kg.search_calls] == ["search_nodes", "search_nodes"]
    assert "Paris" not in caplog.text
    assert "sensitive memory" not in caplog.text
    store = SimpleNamespace(list_snapshots=lambda **kwargs: [])
    (row,) = parse_run_log(log, store=store)
    assert [e["decision"] for e in row["gate_events"]] == ["block", "allow", "allow"]
    first, second = row["tool_calls"]
    if structured:
        assert S_IMODE(trace.stat().st_mode) == 0o600
        attempts = read_traces(trace)["one"]["attempts"]
        assert attempts[0]["kwargs"]["episode_body"] == "sensitive memory"
        assert attempts[0]["status"] == "blocked"
        assert first["args"] == ["London"]
        assert first["kwargs"] == {"max_nodes": 3}
        assert second["kwargs"] == {"query": "Paris"}
        assert second["result"] == [{"name": "node"}]
        prompt = _prompt_for_call(row=row, tool_call=second, call_index=1)
        assert '"Paris"' in prompt and '"node"' in prompt and "Call index:\n2" in prompt
        key = _cache_key(row=row, tool_call=second, call_index=1)
        for edit in ({"kwargs": {"query": "Rome"}}, {"result": []}):
            assert _cache_key(row=row, tool_call={**second, **edit}, call_index=1) != key
    else:
        assert "Legacy console summaries" in _prompt_for_call(
            row=row, tool_call=first, call_index=0
        )
    for call in row["tool_calls"]:
        call["trajectory_state_raw"] = 1
    baseline = {
        **row,
        "variant": "read_only_rlm",
        "runner": "read_only_rlm",
        "run_log_path": "baseline.log",
        "tool_calls": [],
        "tool_call_count": 0,
        "judge_binary": 0,
        "f1": 0,
    }
    winner = build_gate_examples([row, baseline])
    loser = build_gate_examples(
        [{**row, "judge_binary": 0, "f1": 0}, {**baseline, "judge_binary": 1, "f1": 1}]
    )
    assert [e["label"] for e in winner] == [1, 1]
    assert [e["label"] for e in loser] == [0, 0]
    fields = (
        "call_index",
        "prior_call_count",
        "prior_mutating_call_count",
        "prior_same_tool_count",
        "prior_unique_tool_count",
        "writable_flag",
    )
    for examples in (winner, loser, build_trajectory_gate_examples([row])):
        assert [tuple(e[k] for k in fields) for e in examples] == [
            (1, 1, 1, 0, 1, 1),
            (2, 2, 1, 1, 2, 1),
        ]


def test_private_capture_bounds_and_redacts_nested_values():
    from gatekg.utils.trajectory import capture_value

    evidence = capture_value(
        {
            "headers": {"Authorization": "secret", "service_api_key": "private"},
            "body": "x" * 5000,
            "ids": list(range(50)),
        }
    )
    assert evidence["headers"] == {"Authorization": "[REDACTED]", "service_api_key": "[REDACTED]"}
    assert evidence["body"] == {"text": "x" * 4096, "omitted_chars": 904}
    assert evidence["ids"] == {"items": list(range(32)), "omitted_items": 18}


def test_graph_runner_captures_allowed_memory_and_failed_search(tmp_path):
    from gatekg.api.mcp_rlm import _KGMethodMixin
    from gatekg.runners import GraphRLMRunner

    class OfflineGraph(_FakeKG):
        async def search_nodes(self, *args, **kwargs):
            raise ValueError("bad query")

    class OfflineModel(_KGMethodMixin):
        kg = OfflineGraph()

        async def prepare_context(self, *args, **kwargs):
            self.write_group_id = "test-work"
            self.kg.set_active_group(self.write_group_id)

        def predict(self, prompt, question):
            self.add_memory(name="capital", episode_body="Paris is the capital of France.")
            with pytest.raises(ValueError, match="bad query"):
                self.search_nodes(query="London")
            return "Paris"

    path = tmp_path / "run.trajectory.jsonl"
    runner = GraphRLMRunner(
        model=OfflineModel(), prompt=str, rebuild_kg=False, trajectory_path=path
    )
    answer = asyncio.run(runner.predict(ExperimentEntry("one", "Capital?", "Paris", ["fact"])))
    assert answer == "Paris"
    write, search = read_traces(path)["one"]["attempts"]
    assert write["kwargs"]["episode_body"] == "Paris is the capital of France."
    assert write["result"] == {"ok": True}
    assert write["gate_event"] is None
    assert search["status"] == "error" and search["error_type"] == "ValueError"
