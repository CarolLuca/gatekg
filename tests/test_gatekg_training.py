from __future__ import annotations

import json

from gatekg.training.train import split_examples, train_gatekg


def _row(
    *,
    run_log_path: str,
    entry_id: str,
    question: str,
    variant: str,
    f1: float,
    judge_binary: int,
    average_health_delta: float,
    tool_calls: list[dict],
) -> dict:
    return {
        "run_log_path": run_log_path,
        "task_name": "loogle",
        "dataset_variant": "base",
        "entry_index": 0,
        "entry_id": entry_id,
        "question": question,
        "gold": "gold",
        "variant": variant,
        "model_key": "ollama:qwen3.6:27b",
        "kg_model_key": "google:gemini-3-flash-preview",
        "judge_model_key": "google:gemini-3.1-pro-preview",
        "prediction": "pred",
        "judge_raw": str(judge_binary),
        "judge_binary": judge_binary,
        "judge_explanation": "ok",
        "em": 0.0,
        "f1": f1,
        "precision": f1,
        "recall": f1,
        "base_group_id": "mcp-cache_x",
        "write_group_id": "mcp-work_x" if variant != "read_only_rlm" else None,
        "context_chunk_count": 4,
        "kg_context_chunk_count": 4,
        "baseline_snapshot_id": "baseline:mcp-cache_x",
        "run_snapshot_id": "run:x" if variant != "read_only_rlm" else None,
        "baseline_average_health": 0.8,
        "run_average_health": 0.8 + average_health_delta,
        "average_health_delta": average_health_delta,
        "baseline_metric_scores": {"coverage": 0.7, "consistency": 0.8},
        "run_metric_scores": {"coverage": 0.7 + average_health_delta, "consistency": 0.8},
        "metric_deltas": {"coverage": average_health_delta, "consistency": 0.0},
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "mutating_tool_call_count": sum(1 for call in tool_calls if call["is_mutating"]),
        "artifact_quality": "complete",
        "parser_version": "gatekg-v1",
    }


def test_train_gatekg_writes_expected_artifacts(tmp_path):
    rows = [
        _row(
            run_log_path="run_a_read_only.log",
            entry_id="entry-a",
            question="Question A",
            variant="read_only_rlm",
            f1=0.2,
            judge_binary=0,
            average_health_delta=0.0,
            tool_calls=[
                {"tool": "search_nodes", "is_mutating": False},
                {"tool": "search_memory_facts", "is_mutating": False},
            ],
        ),
        _row(
            run_log_path="run_a_read_write.log",
            entry_id="entry-a",
            question="Question A",
            variant="read_write_rlm",
            f1=0.9,
            judge_binary=1,
            average_health_delta=0.2,
            tool_calls=[
                {"tool": "search_nodes", "is_mutating": False},
                {"tool": "add_memory", "is_mutating": True},
            ],
        ),
        _row(
            run_log_path="run_a_aggressive.log",
            entry_id="entry-a",
            question="Question A",
            variant="aggressive_write_rlm",
            f1=0.4,
            judge_binary=0,
            average_health_delta=-0.1,
            tool_calls=[
                {"tool": "search_nodes", "is_mutating": False},
                {"tool": "add_triplet", "is_mutating": True},
            ],
        ),
        _row(
            run_log_path="run_b_read_only.log",
            entry_id="entry-b",
            question="Question B",
            variant="read_only_rlm",
            f1=0.8,
            judge_binary=1,
            average_health_delta=0.0,
            tool_calls=[
                {"tool": "search_nodes", "is_mutating": False},
            ],
        ),
        _row(
            run_log_path="run_b_read_write.log",
            entry_id="entry-b",
            question="Question B",
            variant="read_write_rlm",
            f1=0.5,
            judge_binary=0,
            average_health_delta=0.1,
            tool_calls=[
                {"tool": "search_nodes", "is_mutating": False},
                {"tool": "add_memory", "is_mutating": True},
            ],
        ),
        _row(
            run_log_path="run_b_aggressive.log",
            entry_id="entry-b",
            question="Question B",
            variant="aggressive_write_rlm",
            f1=0.3,
            judge_binary=0,
            average_health_delta=-0.2,
            tool_calls=[
                {"tool": "search_nodes", "is_mutating": False},
                {"tool": "add_triplet", "is_mutating": True},
            ],
        ),
    ]
    dataset_path = tmp_path / "loogle_mcp_runs.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    artifact_dir = tmp_path / "artifacts"
    metrics = train_gatekg(
        dataset_path=dataset_path,
        artifact_dir=artifact_dir,
        selector_epochs=1,
        gate_epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        seed=0,
    )

    assert metrics["dataset"]["selector_example_count"] == 2
    assert metrics["dataset"]["gate_example_count"] > 0
    assert (artifact_dir / "selector.pt").is_file()
    assert (artifact_dir / "tool_gate.pt").is_file()
    assert (artifact_dir / "feature_config.json").is_file()
    assert (artifact_dir / "label_maps.json").is_file()
    assert (artifact_dir / "training_metrics.json").is_file()
    assert (artifact_dir / "training_results.json").is_file()
    assert (artifact_dir / "training_log.jsonl").is_file()
    assert (artifact_dir / "training_summary.md").is_file()
    assert "loss" in metrics["selector"]["train"]
    assert "generalization_gap" in metrics["selector"]
    from gatekg.training.features import load_inference_bundle

    bundle = load_inference_bundle(artifact_dir)
    assert set(bundle.label_maps.variant) == {"read_only_rlm", "read_write_rlm"}
    assert bundle.feature_config.selector_dropout == 0.2
    assert bundle.feature_config.gate_dropout == 0.1
    for model in (bundle.selector_model, bundle.tool_gate_model):
        assert not model.training
        for key in model.state_dict():
            assert "dataset_variant_embed" not in key
            assert "model_key_embed" not in key


def test_split_examples_grouped_keeps_entry_ids_disjoint():
    examples = []
    for group_index in range(10):
        for duplicate_index in range(2):
            examples.append(
                {
                    "dataset_variant": "base",
                    "model_key": "model",
                    "entry_id": f"entry-{group_index}",
                    "duplicate": duplicate_index,
                }
            )

    split = split_examples(examples, seed=0, group_by_identity=True)

    train_keys = {
        (example["dataset_variant"], example["model_key"], example["entry_id"])
        for example in split.train
    }
    val_keys = {
        (example["dataset_variant"], example["model_key"], example["entry_id"])
        for example in split.val
    }
    assert train_keys
    assert val_keys
    assert train_keys.isdisjoint(val_keys)


def test_train_gatekg_supports_trajectory_gate_labels(tmp_path):
    rows = [
        _row(
            run_log_path="run_a_read_only.log",
            entry_id="entry-a",
            question="Question A",
            variant="read_only_rlm",
            f1=0.2,
            judge_binary=0,
            average_health_delta=0.0,
            tool_calls=[
                {
                    "tool": "search_nodes",
                    "is_mutating": False,
                    "trajectory_state_raw": -1,
                },
            ],
        ),
        _row(
            run_log_path="run_a_read_write.log",
            entry_id="entry-a",
            question="Question A",
            variant="read_write_rlm",
            f1=0.9,
            judge_binary=1,
            average_health_delta=0.2,
            tool_calls=[
                {
                    "tool": "search_nodes",
                    "is_mutating": False,
                    "trajectory_state_raw": 0,
                },
                {
                    "tool": "add_memory",
                    "is_mutating": True,
                    "trajectory_state_raw": 1,
                },
            ],
        ),
        _row(
            run_log_path="run_b_read_only.log",
            entry_id="entry-b",
            question="Question B",
            variant="read_only_rlm",
            f1=0.8,
            judge_binary=1,
            average_health_delta=0.0,
            tool_calls=[
                {
                    "tool": "search_nodes",
                    "is_mutating": False,
                    "trajectory_state_raw": 1,
                },
            ],
        ),
        _row(
            run_log_path="run_b_read_write.log",
            entry_id="entry-b",
            question="Question B",
            variant="read_write_rlm",
            f1=0.5,
            judge_binary=0,
            average_health_delta=0.1,
            tool_calls=[
                {
                    "tool": "search_nodes",
                    "is_mutating": False,
                    "trajectory_state_raw": 0,
                },
                {
                    "tool": "add_memory",
                    "is_mutating": True,
                    "trajectory_state_raw": -1,
                },
            ],
        ),
    ]
    dataset_path = tmp_path / "trajectory_dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    artifact_dir = tmp_path / "trajectory_artifacts"
    metrics = train_gatekg(
        dataset_path=dataset_path,
        artifact_dir=artifact_dir,
        selector_epochs=1,
        gate_epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        seed=0,
        weight_decay=1e-2,
        early_stopping_patience=1,
        group_split=True,
        selector_dropout=0.1,
        gate_dropout=0.1,
        gate_label_source="trajectory_state",
    )

    assert metrics["training"]["gate_label_source"] == "trajectory_state"
    assert metrics["training"]["group_split"] is True
    assert metrics["dataset"]["gate_example_count"] == 6
    assert (artifact_dir / "tool_gate.pt").is_file()


def test_default_partition_groups_question_across_models_and_context_variants():
    from gatekg.training.train import parse_args

    assert parse_args([]).group_split
    rows = [
        dict(
            question=f"What is {i}?",
            entry_id=f"{model}-{i}",
            model_key=model,
            dataset_variant=model,
        )
        for i in range(10)
        for model in ("a", "b")
    ]
    split = split_examples(rows, seed=1)
    assert len(split.val) == 4
    assert {row["question"] for row in split.val}.isdisjoint(row["question"] for row in split.train)


def test_reward_labels_require_completed_evaluation():
    import pytest

    from gatekg.training.dataset import annotate_rewards

    for row in (
        {"judge_binary": None},
        {"judge_binary": 0, "prediction": "Server error"},
        {"judge_binary": 0, "judge_explanation": "Judge request failed after retries. offline"},
    ):
        with pytest.raises(ValueError, match="Missing or failed evaluation"):
            annotate_rewards([row])
