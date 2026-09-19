from __future__ import annotations

import json


def test_advance_gatekg_round_writes_manifest(monkeypatch, tmp_path):
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
            "tool_calls": [
                {
                    "tool": "search_nodes",
                    "is_mutating": False,
                    "trajectory_state_raw": 1,
                    "trajectory_state": 1,
                }
            ],
            "tool_call_count": 1,
            "mutating_tool_call_count": 0,
            "artifact_quality": "complete",
            "parser_version": "gatekg-v1",
        }
    ]

    def fake_label_dataset(**kwargs):
        advance.write_dataset(
            rows,
            output_path=kwargs["output_path"],
            summary_path=kwargs["summary_path"],
        )
        return rows

    monkeypatch.setattr(advance, "build_dataset_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(advance, "label_dataset", fake_label_dataset)
    monkeypatch.setattr(
        advance,
        "train_gatekg",
        lambda **kwargs: {
            "dataset": {"gate_example_count": 1},
            "training": {
                "gate_label_source": kwargs["gate_label_source"],
                "group_split": kwargs["group_split"],
            },
            "selector": {
                "train": {"accuracy": 1.0, "loss": 0.1},
                "val": {"accuracy": 1.0, "loss": 0.1},
            },
            "tool_gate": {
                "train": {"accuracy": 1.0, "loss": 0.1},
                "val": {"accuracy": 1.0, "loss": 0.1},
            },
        },
    )

    artifact_dir = tmp_path / "artifacts" / "round_001"
    exit_code = advance.main(
        [
            "--gate-label-source",
            "trajectory_state",
            "--round",
            "1",
            "--output-root",
            str(tmp_path / "rounds"),
            "--artifact-dir",
            str(artifact_dir),
        ]
    )

    assert exit_code == 0
    manifest_path = artifact_dir / "round_manifest.json"
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["round"] == 1
    assert manifest["labeled_row_count"] == 1
    assert manifest["training_metrics"]["training"]["gate_label_source"] == "trajectory_state"
    assert manifest["training_metrics"]["training"]["group_split"] is True
