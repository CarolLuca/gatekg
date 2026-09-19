from __future__ import annotations

import json


def test_interpret_gatekg_dataset_and_run_reports(monkeypatch, tmp_path):
    import gatekg.training.interpret as interpret

    dataset_rows = [
        {
            "run_log_path": "reference_a.log",
            "task_name": "loogle",
            "dataset_variant": "base",
            "entry_index": 0,
            "entry_id": "entry-a",
            "question": "Question A",
            "gold": "gold",
            "variant": "read_write_rlm",
            "model_key": "ollama:qwen3.6:27b",
            "kg_model_key": "google:gemini-3-flash-preview",
            "judge_model_key": "google:gemini-3.1-pro-preview",
            "prediction": "pred",
            "judge_raw": "1",
            "judge_binary": 1,
            "judge_explanation": "ok",
            "em": 0.0,
            "f1": 0.9,
            "precision": 0.9,
            "recall": 0.9,
            "base_group_id": "mcp-cache_a",
            "write_group_id": "mcp-work_a",
            "context_chunk_count": 4,
            "kg_context_chunk_count": 4,
            "baseline_snapshot_id": "baseline:mcp-cache_a",
            "run_snapshot_id": "run:a",
            "baseline_average_health": 0.8,
            "run_average_health": 0.9,
            "average_health_delta": 0.1,
            "baseline_metric_scores": {"coverage": 0.7},
            "run_metric_scores": {"coverage": 0.8},
            "metric_deltas": {"coverage": 0.1},
            "tool_calls": [
                {"tool": "search_nodes", "is_mutating": False},
                {"tool": "add_memory", "is_mutating": True},
            ],
            "tool_call_count": 2,
            "mutating_tool_call_count": 1,
            "artifact_quality": "complete",
            "parser_version": "gatekg-v1",
        },
        {
            "run_log_path": "reference_a_ro.log",
            "task_name": "loogle",
            "dataset_variant": "base",
            "entry_index": 0,
            "entry_id": "entry-a",
            "question": "Question A",
            "gold": "gold",
            "variant": "read_only_rlm",
            "model_key": "ollama:qwen3.6:27b",
            "kg_model_key": "google:gemini-3-flash-preview",
            "judge_model_key": "google:gemini-3.1-pro-preview",
            "prediction": "pred",
            "judge_raw": "0",
            "judge_binary": 0,
            "judge_explanation": "ok",
            "em": 0.0,
            "f1": 0.3,
            "precision": 0.3,
            "recall": 0.3,
            "base_group_id": "mcp-cache_a",
            "write_group_id": None,
            "context_chunk_count": 4,
            "kg_context_chunk_count": 4,
            "baseline_snapshot_id": "baseline:mcp-cache_a",
            "run_snapshot_id": None,
            "baseline_average_health": 0.8,
            "run_average_health": 0.8,
            "average_health_delta": 0.0,
            "baseline_metric_scores": {"coverage": 0.7},
            "run_metric_scores": {"coverage": 0.7},
            "metric_deltas": {"coverage": 0.0},
            "tool_calls": [{"tool": "search_nodes", "is_mutating": False}],
            "tool_call_count": 1,
            "mutating_tool_call_count": 0,
            "artifact_quality": "complete",
            "parser_version": "gatekg-v1",
        },
    ]

    dataset_path = tmp_path / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in dataset_rows:
            handle.write(json.dumps(row) + "\n")

    dataset_prefix = tmp_path / "dataset_report"
    assert (
        interpret.main(
            [
                "dataset",
                "--dataset",
                str(dataset_path),
                "--output-prefix",
                str(dataset_prefix),
            ]
        )
        == 0
    )
    assert dataset_prefix.with_suffix(".json").is_file()
    assert dataset_prefix.with_suffix(".md").is_file()

    run_log_path = tmp_path / "run.log"
    run_log_path.write_text(
        "\n".join(
            [
                "Testing entry 1/1",
                '2026-04-24 12:00:00 | INFO | api.gatekg | GateKG selection event: {"dataset_variant": "base", "question": "Question A", "selected_variant": "read_write_rlm", "selector_scores": {"read_only_rlm": 0.1, "read_write_rlm": 0.9}, "base_group_id": "mcp-cache_a", "write_group_id": "mcp-work_a", "baseline_average_health": 0.8}',
                '2026-04-24 12:00:01 | INFO | api.gatekg | GateKG gate event: {"call_index": 0, "decision": "allow", "tool": "search_nodes", "score": 0.9, "selected_variant": "read_write_rlm"}',
                '2026-04-24 12:00:02 | INFO | api.gatekg | GateKG gate event: {"call_index": 1, "decision": "block", "tool": "add_memory", "score": 0.1, "selected_variant": "read_write_rlm"}',
                '2026-04-24 12:00:03 | INFO | api.gatekg | GateKG run summary event: {"question": "Question A", "dataset_variant": "base", "selected_variant": "read_write_rlm", "allowed_tool_count": 1, "blocked_tool_count": 1, "run_snapshot_id": "run:new"}',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        interpret,
        "parse_run_log",
        lambda log_path: [
            {
                **dataset_rows[0],
                "run_log_path": str(log_path),
                "prediction": "pred",
            }
        ],
    )

    run_prefix = tmp_path / "run_report"
    assert (
        interpret.main(
            [
                "run",
                "--log",
                str(run_log_path),
                "--dataset",
                str(dataset_path),
                "--output-prefix",
                str(run_prefix),
            ]
        )
        == 0
    )
    assert run_prefix.with_suffix(".json").is_file()
    assert run_prefix.with_suffix(".md").is_file()

    payload = json.loads(run_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["selected_variant_counts"]["read_write_rlm"] == 1
    assert payload["gate_decision_counts"]["block"] == 1
