from __future__ import annotations

import json


def test_advance_gatekg_round_writes_round_manifest(monkeypatch, tmp_path):
    import gatekg.training.rounds as advance

    rows = [
        {
            "run_log_path": "run.log",
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
            "tool_calls": [{"tool": "search_nodes", "is_mutating": False}],
            "tool_call_count": 1,
            "mutating_tool_call_count": 0,
            "artifact_quality": "complete",
            "parser_version": "gatekg-v1",
        }
    ]

    monkeypatch.setattr(advance, "build_dataset_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(
        advance,
        "train_gatekg",
        lambda **kwargs: {
            "dataset": {"selector_example_count": 1},
            "training": {"seed": kwargs["seed"]},
            "selector": {"train": {"accuracy": 1.0}},
            "tool_gate": {"train": {"accuracy": 1.0}},
        },
    )

    exit_code = advance.main(
        [
            "--round",
            "2",
            "--output-root",
            str(tmp_path / "rounds"),
            "--artifact-dir",
            str(tmp_path / "artifacts" / "round_002"),
            "--dataset-output",
            str(tmp_path / "rounds" / "round_002" / "dataset.jsonl"),
            "--summary-output",
            str(tmp_path / "rounds" / "round_002" / "dataset.summary.json"),
        ]
    )

    assert exit_code == 0
    artifact_dir = tmp_path / "artifacts" / "round_002"
    manifest_path = artifact_dir / "round_manifest.json"
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["round"] == 2
    assert manifest["row_count"] == 1
    assert manifest["training_metrics"]["training"]["seed"] == 0
