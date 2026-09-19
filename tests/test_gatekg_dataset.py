from __future__ import annotations

from pathlib import Path

from gatekg.training.dataset import (
    ExperimentSpec,
    build_trajectory_gate_examples,
    extract_result_blocks,
    fill_row_trajectory_states,
    parse_run_log,
    resolve_experiment_spec,
    right_fill_trajectory_states,
    split_execution_segments,
)
from gatekg.utils.eval_runner import ExperimentEntry


def test_split_execution_segments_supports_context_counts():
    log_text = """
Testing entry 1/2
2026-04-24 12:00:00 | INFO | api.mcp_rlm | Preparing context: runtime_chunks=4 kg_chunks=4 cache_model_key=google:gemini-3-flash-preview base_group_id=mcp-cache_readonly123 writable=False rebuild=False
2026-04-24 12:00:00 | INFO | api.mcp_rlm | KG tool call: search_nodes args=('alpha',) kwargs=dict[keys=['max_nodes']]
Testing entry 2/2
2026-04-24 12:01:00 | INFO | api.mcp_rlm | Preparing context: runtime_chunks=5 kg_chunks=3 cache_model_key=google:gemini-3-flash-preview base_group_id=mcp-cache_current123 writable=True rebuild=False
2026-04-24 12:01:00 | INFO | api.mcp_rlm | Writable run configured with write_group_id=mcp-work_current123
2026-04-24 12:01:00 | INFO | api.mcp_rlm | KG tool call: add_memory args=() kwargs=dict[keys=['name']]
"""
    segments = split_execution_segments(log_text)

    assert len(segments) == 2
    assert segments[0].context_chunk_count == 4
    assert segments[0].kg_context_chunk_count == 4
    assert segments[0].base_group_id == "mcp-cache_readonly123"
    assert segments[1].context_chunk_count == 5
    assert segments[1].kg_context_chunk_count == 3
    assert segments[1].write_group_id == "mcp-work_current123"
    assert segments[1].tool_calls[0]["tool"] == "add_memory"
    assert segments[1].tool_calls[0]["is_mutating"] is True


def test_extract_result_blocks_reads_binary_judge_scores():
    log_text = """
-----------------
Question text
Gold answer
-----------------
Predicted answer
-----------------
judge score: 1
judge explanation: Strong match.
-----------------
curr judge: 1.0
curr em: 0.0
curr f1: 0.5
-----------------
"""
    blocks = extract_result_blocks(log_text)

    assert len(blocks) == 1
    assert blocks[0].prediction == "Predicted answer"
    assert blocks[0].judge_raw == "1"
    assert blocks[0].judge_binary == 1
    assert blocks[0].judge_explanation == "Strong match."


def test_parse_run_log_joins_health_snapshots(monkeypatch, tmp_path):
    log_path = tmp_path / "logs" / "loogle" / "read_only_rlm" / "ollama_qwen" / "run_1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                "Task: loogle",
                "Model: ollama:qwen3.6:27b",
                "KG cache key: google:gemini-3-flash-preview",
                "Judge model: gemini-3.1-pro-preview",
                "Testing entry 1/1",
                "2026-04-24 12:00:00 | INFO | api.mcp_rlm | Preparing context: runtime_chunks=4 kg_chunks=4 cache_model_key=google:gemini-3-flash-preview base_group_id=mcp-cache_health123 writable=False rebuild=False",
                "2026-04-24 12:00:00 | INFO | api.mcp_rlm | KG tool call: search_nodes args=('alpha',) kwargs=dict[keys=['max_nodes']]",
                "-----------------",
                "Question text",
                "Gold answer",
                "-----------------",
                "Gold answer",
                "-----------------",
                "judge score: 1",
                "judge explanation: Exact match.",
                "-----------------",
                "curr judge: 1.0",
                "curr em: 1.0",
                "curr f1: 1.0",
                "-----------------",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "gatekg.training.dataset.load_entries_for_log",
        lambda *args, **kwargs: (
            ExperimentSpec(
                task_name="loogle",
                log_family="read_only_rlm",
                variant="read_only_rlm",
                dataset_variant="base",
                dataset_path=Path("data/loogle/longdep_qa.jsonl"),
            ),
            [
                ExperimentEntry(
                    entry_id="entry-1",
                    question="Question text",
                    gold="Gold answer",
                    context_chunks=["a", "b"],
                )
            ],
        ),
    )

    class FakeStore:
        def get_baseline_snapshot(self, base_group_id):
            assert base_group_id == "mcp-cache_health123"
            return {
                "snapshot_id": "baseline:mcp-cache_health123",
                "average_health_score": 0.8,
                "metric_scores": {"coverage": 0.7},
            }

        def list_snapshots(self, **kwargs):
            return []

    rows = parse_run_log(log_path, store=FakeStore())

    assert len(rows) == 1
    row = rows[0]
    assert row["variant"] == "read_only_rlm"
    assert row["judge_binary"] == 1
    assert row["baseline_snapshot_id"] == "baseline:mcp-cache_health123"
    assert row["baseline_average_health"] == 0.8
    assert row["average_health_delta"] == 0.0
    assert row["tool_call_count"] == 1


def test_resolve_experiment_spec_uses_portable_manifest(tmp_path):
    from gatekg.datasets import DatasetSpec
    from gatekg.records import write_run_manifest
    from gatekg.training.dataset import load_entries_for_log

    log_path = tmp_path / "run.log"
    entry = ExperimentEntry("one", "Question", "Answer", ["Context"])
    write_run_manifest(
        log_path,
        run_id="run",
        dataset=DatasetSpec("custom", tmp_path / "missing.jsonl"),
        runner="read_only_rlm",
        judge="exact",
        entries=[entry],
    )
    spec = resolve_experiment_spec(log_path)
    assert spec.task_name == "custom"
    assert spec.variant == "read_only_rlm"
    assert load_entries_for_log(log_path)[1] == [entry]


def test_right_fill_trajectory_states_uses_next_non_zero_and_fallback():
    assert right_fill_trajectory_states([0, 0, 1, 0, -1, 0], fallback_state=-1) == [
        1,
        1,
        1,
        -1,
        -1,
        -1,
    ]
    assert right_fill_trajectory_states([0, 0], fallback_state=1) == [1, 1]


def test_build_trajectory_gate_examples_uses_filled_states():
    row = {
        "run_log_path": "run.log",
        "task_name": "loogle",
        "dataset_variant": "base",
        "entry_index": 0,
        "entry_id": "entry-a",
        "question": "Question A",
        "gold": "gold",
        "variant": "read_write_rlm",
        "model_key": "ollama:qwen3.6:27b",
        "judge_binary": 0,
        "f1": 0.0,
        "average_health_delta": 0.0,
        "tool_calls": [
            {"tool": "search_nodes", "is_mutating": False, "trajectory_state_raw": 0},
            {
                "tool": "search_memory_facts",
                "is_mutating": False,
                "trajectory_state_raw": 1,
            },
            {"tool": "add_memory", "is_mutating": True, "trajectory_state_raw": 0},
        ],
        "tool_call_count": 3,
        "mutating_tool_call_count": 1,
        "baseline_average_health": 0.8,
        "baseline_metric_scores": {"coverage": 0.7},
    }

    filled_row = fill_row_trajectory_states(row)
    assert [call["trajectory_state"] for call in filled_row["tool_calls"]] == [1, 1, -1]

    examples = build_trajectory_gate_examples([row])
    assert [example["label"] for example in examples] == [1, 1, 0]
    assert [example["trajectory_state"] for example in examples] == [1, 1, -1]


def test_composite_reward_formula():
    import pytest

    from gatekg.training.dataset import annotate_rewards

    rows = [
        {
            "run_log_path": "run_1.log",
            "dataset_variant": "base",
            "model_key": "google:gemini-3-flash-preview",
            "variant": "read_write_rlm",
            "entry_id": "entry-1",
            "judge_binary": 1,
            "f1": 0.8,
            "average_health_delta": 0.2,
            "tool_call_count": 5,
            "mutating_tool_call_count": 2,
        },
        {
            "run_log_path": "run_2.log",
            "dataset_variant": "base",
            "model_key": "google:gemini-3-flash-preview",
            "variant": "read_write_rlm",
            "entry_id": "entry-2",
            "judge_binary": 0,
            "f1": 0.4,
            "average_health_delta": -0.1,
            "tool_call_count": 10,
            "mutating_tool_call_count": 5,
        },
    ]

    annotated = annotate_rewards(rows)
    assert pytest.approx(annotated[0]["composite_reward"]) == 0.97
    assert pytest.approx(annotated[1]["composite_reward"]) == 0.06
