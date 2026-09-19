"""Training entry point."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gatekg.paths import policy_artifact_dir, runtime_paths

PROJECT_ROOT = runtime_paths().root

from gatekg.training.dataset import (
    build_gate_examples,
    build_selector_targets,
    build_trajectory_gate_examples,
    read_dataset_rows,
)
from gatekg.training.features import (
    GATEKG_VARIANTS,
    FeatureConfig,
    LabelMaps,
    as_float_tensor,
    build_feature_config,
    build_gate_model,
    build_label_maps,
    build_selector_model,
    encode_input_label,
    gate_numeric_vector,
    normalize_variant_name,
    save_artifact_metadata,
    selector_numeric_vector,
    tensorize_question_batch,
)

DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "gatekg" / "runs.jsonl"
DEFAULT_SELECTOR_EPOCHS = 25
DEFAULT_GATE_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 3e-3
DEFAULT_SEED = 0
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_SELECTOR_DROPOUT = 0.2
DEFAULT_GATE_DROPOUT = 0.1


@dataclass(frozen=True)
class TrainEvalSplit:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the GateKG selector and tool gate models.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Canonical GateKG dataset JSONL path.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(policy_artifact_dir()),
        help="Directory where trained selector/gate artifacts will be written.",
    )
    parser.add_argument(
        "--selector-epochs",
        type=int,
        default=DEFAULT_SELECTOR_EPOCHS,
        help="Number of selector training epochs.",
    )
    parser.add_argument(
        "--gate-epochs",
        type=int,
        default=DEFAULT_GATE_EPOCHS,
        help="Number of tool-gate training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size for both models.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic random seed.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Stop after this many non-improving validation epochs. Disabled at 0.",
    )
    parser.add_argument(
        "--group-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep calls and trajectories for the same question together.",
    )
    parser.add_argument(
        "--selector-dropout",
        type=float,
        default=DEFAULT_SELECTOR_DROPOUT,
        help="Dropout probability for the selector/router head.",
    )
    parser.add_argument(
        "--gate-dropout",
        type=float,
        default=DEFAULT_GATE_DROPOUT,
        help="Dropout probability for the tool-gate head.",
    )
    parser.add_argument(
        "--gate-label-source",
        choices=("reward_divergence", "trajectory_state"),
        default="reward_divergence",
        help="Training target source for the tool gate.",
    )
    parser.add_argument(
        "--include-aggressive-write",
        action="store_true",
        help="Train the optional three-mode policy.",
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def question_identity(example: dict[str, Any]) -> str:
    question = " ".join(str(example.get("question") or "").casefold().split())
    return question or str(example["entry_id"])


def split_examples(
    examples: Sequence[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float = 0.2,
    group_by_identity: bool = True,
) -> TrainEvalSplit:
    materialized_examples = [dict(example) for example in examples]
    if len(materialized_examples) < 5:
        return TrainEvalSplit(train=materialized_examples, val=[])

    rng = random.Random(seed)
    if group_by_identity:
        groups: dict[str, list[dict[str, Any]]] = {}
        for example in materialized_examples:
            group_key = question_identity(example)
            groups.setdefault(group_key, []).append(example)

        group_items = sorted(groups.items())
        if len(group_items) < 5:
            return TrainEvalSplit(train=materialized_examples, val=[])
        rng.shuffle(group_items)
        val_group_count = max(1, int(len(group_items) * val_fraction))
        if val_group_count >= len(group_items):
            val_group_count = len(group_items) - 1
        val_keys = {group_key for group_key, _ in group_items[:val_group_count]}
        train: list[dict[str, Any]] = []
        val: list[dict[str, Any]] = []
        for group_key, group_examples in group_items:
            if group_key in val_keys:
                val.extend(group_examples)
            else:
                train.extend(group_examples)
        return TrainEvalSplit(train=train, val=val)

    rng.shuffle(materialized_examples)
    val_count = max(1, int(len(materialized_examples) * val_fraction))
    if val_count >= len(materialized_examples):
        val_count = len(materialized_examples) - 1
    return TrainEvalSplit(
        train=materialized_examples[val_count:],
        val=materialized_examples[:val_count],
    )


def _selector_batch(
    examples: Sequence[dict[str, Any]],
    *,
    feature_config: FeatureConfig,
    label_maps: LabelMaps,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids, offsets = tensorize_question_batch(
        [str(example.get("question") or "") for example in examples],
        buckets=feature_config.token_hash_buckets,
        device=device,
    )
    dataset_variant_ids = torch.tensor(
        [
            encode_input_label(
                label_maps.dataset_variant,
                str(example.get("dataset_variant") or ""),
            )
            for example in examples
        ],
        dtype=torch.long,
        device=device,
    )
    model_key_ids = torch.tensor(
        [
            encode_input_label(
                label_maps.model_key,
                str(example.get("model_key") or ""),
            )
            for example in examples
        ],
        dtype=torch.long,
        device=device,
    )
    numeric_features = as_float_tensor(
        [
            selector_numeric_vector(
                example,
                health_metric_names=feature_config.health_metric_names,
            )
            for example in examples
        ],
        device=device,
    )
    labels = torch.tensor(
        [
            label_maps.variant[
                normalize_variant_name(
                    str(example.get("target_variant") or example.get("variant") or "")
                )
            ]
            for example in examples
        ],
        dtype=torch.long,
        device=device,
    )
    return token_ids, offsets, dataset_variant_ids, model_key_ids, numeric_features, labels


def _gate_batch(
    examples: Sequence[dict[str, Any]],
    *,
    feature_config: FeatureConfig,
    label_maps: LabelMaps,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    token_ids, offsets = tensorize_question_batch(
        [str(example.get("question") or "") for example in examples],
        buckets=feature_config.token_hash_buckets,
        device=device,
    )
    dataset_variant_ids = torch.tensor(
        [
            encode_input_label(
                label_maps.dataset_variant,
                str(example.get("dataset_variant") or ""),
            )
            for example in examples
        ],
        dtype=torch.long,
        device=device,
    )
    model_key_ids = torch.tensor(
        [
            encode_input_label(
                label_maps.model_key,
                str(example.get("model_key") or ""),
            )
            for example in examples
        ],
        dtype=torch.long,
        device=device,
    )
    variant_ids = torch.tensor(
        [
            label_maps.variant[normalize_variant_name(str(example.get("variant") or ""))]
            for example in examples
        ],
        dtype=torch.long,
        device=device,
    )
    tool_ids = torch.tensor(
        [
            encode_input_label(
                label_maps.tool_name,
                str(example.get("tool_name") or ""),
            )
            for example in examples
        ],
        dtype=torch.long,
        device=device,
    )
    numeric_features = as_float_tensor(
        [
            gate_numeric_vector(
                example,
                health_metric_names=feature_config.health_metric_names,
                call_index=int(example.get("call_index") or 0),
                prior_call_count=int(example.get("prior_call_count") or 0),
                prior_mutating_call_count=int(example.get("prior_mutating_call_count") or 0),
                prior_same_tool_count=int(example.get("prior_same_tool_count") or 0),
                prior_unique_tool_count=int(example.get("prior_unique_tool_count") or 0),
                writable_flag=int(example.get("writable_flag") or 0),
            )
            for example in examples
        ],
        device=device,
    )
    labels = torch.tensor(
        [float(example.get("label") or 0.0) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    return (
        token_ids,
        offsets,
        dataset_variant_ids,
        model_key_ids,
        variant_ids,
        tool_ids,
        numeric_features,
        labels,
    )


def _batched(sequence: Sequence[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [
        list(sequence[index : index + batch_size]) for index in range(0, len(sequence), batch_size)
    ]


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _log_epoch(
    *,
    component: str,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    best_epoch: int,
) -> None:
    width = 24
    completed = min(width, int(width * epoch / total_epochs)) if total_epochs else width
    bar = "[" + ("#" * completed) + ("-" * (width - completed)) + "]"
    print(
        f"[{component}] {bar} epoch={epoch}/{total_epochs} "
        f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
        f"best_epoch={best_epoch}",
        flush=True,
    )


def selector_loss(
    model: nn.Module,
    examples: Sequence[dict[str, Any]],
    *,
    feature_config: FeatureConfig,
    label_maps: LabelMaps,
    batch_size: int,
    device: torch.device,
) -> float:
    if not examples:
        return 0.0
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in _batched(examples, batch_size):
            (
                token_ids,
                offsets,
                dataset_variant_ids,
                model_key_ids,
                numeric_features,
                labels,
            ) = _selector_batch(
                batch,
                feature_config=feature_config,
                label_maps=label_maps,
                device=device,
            )
            logits = model(
                token_ids,
                offsets,
                dataset_variant_ids,
                model_key_ids,
                numeric_features,
            )
            total_loss += float(loss_fn(logits, labels).item())
            total_count += len(batch)
    return float(total_loss / total_count) if total_count else 0.0


def gate_loss(
    model: nn.Module,
    examples: Sequence[dict[str, Any]],
    *,
    feature_config: FeatureConfig,
    label_maps: LabelMaps,
    batch_size: int,
    device: torch.device,
) -> float:
    if not examples:
        return 0.0
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in _batched(examples, batch_size):
            (
                token_ids,
                offsets,
                dataset_variant_ids,
                model_key_ids,
                variant_ids,
                tool_ids,
                numeric_features,
                labels,
            ) = _gate_batch(
                batch,
                feature_config=feature_config,
                label_maps=label_maps,
                device=device,
            )
            logits = model(
                token_ids,
                offsets,
                dataset_variant_ids,
                model_key_ids,
                variant_ids,
                tool_ids,
                numeric_features,
            )
            total_loss += float(loss_fn(logits, labels).item())
            total_count += len(batch)
    return float(total_loss / total_count) if total_count else 0.0


def train_selector(
    split: TrainEvalSplit,
    *,
    feature_config: FeatureConfig,
    label_maps: LabelMaps,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    print(
        "Training selector/router: "
        f"train={len(split.train)} val={len(split.val)} "
        f"epochs={epochs} batch_size={batch_size} "
        f"lr={learning_rate} weight_decay={weight_decay}",
        flush=True,
    )
    model = build_selector_model(feature_config, label_maps).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_state = _cpu_state_dict(model)
    best_epoch = 0
    best_score = float("inf")
    non_improving_epochs = 0
    stopped_epoch = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in _batched(split.train, batch_size):
            optimizer.zero_grad()
            (
                token_ids,
                offsets,
                dataset_variant_ids,
                model_key_ids,
                numeric_features,
                labels,
            ) = _selector_batch(
                batch,
                feature_config=feature_config,
                label_maps=label_maps,
                device=device,
            )
            logits = model(
                token_ids,
                offsets,
                dataset_variant_ids,
                model_key_ids,
                numeric_features,
            )
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            total_count += len(batch)
        train_loss = (total_loss / total_count) if total_count else 0.0
        val_loss = selector_loss(
            model,
            split.val,
            feature_config=feature_config,
            label_maps=label_maps,
            batch_size=batch_size,
            device=device,
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        score = val_loss if split.val else train_loss
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = _cpu_state_dict(model)
            non_improving_epochs = 0
        else:
            non_improving_epochs += 1
            if (
                early_stopping_patience > 0
                and split.val
                and non_improving_epochs >= early_stopping_patience
            ):
                stopped_epoch = epoch + 1
                _log_epoch(
                    component="selector",
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_epoch=best_epoch,
                )
                break
        _log_epoch(
            component="selector",
            epoch=epoch + 1,
            total_epochs=epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            best_epoch=best_epoch,
        )

    model.load_state_dict(best_state)

    metrics = evaluate_selector(
        model,
        split,
        feature_config=feature_config,
        label_maps=label_maps,
        batch_size=batch_size,
        device=device,
    )
    metrics["history"] = history
    metrics["best_epoch"] = float(best_epoch)
    metrics["stopped_epoch"] = float(stopped_epoch)
    metrics["early_stopped"] = bool(stopped_epoch)
    metrics["generalization_gap"] = float(
        metrics["train"]["accuracy"] - metrics["val"]["accuracy"]
        if metrics["val"]["count"]
        else 0.0
    )
    print(
        "Finished selector/router: "
        f"train_acc={metrics['train']['accuracy']:.4f} "
        f"val_acc={metrics['val']['accuracy']:.4f} "
        f"train_loss={metrics['train']['loss']:.6f} "
        f"val_loss={metrics['val']['loss']:.6f} "
        f"best_epoch={metrics['best_epoch']:.0f}",
        flush=True,
    )
    return model, metrics


def train_gate(
    split: TrainEvalSplit,
    *,
    feature_config: FeatureConfig,
    label_maps: LabelMaps,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    print(
        "Training tool gate: "
        f"train={len(split.train)} val={len(split.val)} "
        f"epochs={epochs} batch_size={batch_size} "
        f"lr={learning_rate} weight_decay={weight_decay}",
        flush=True,
    )
    model = build_gate_model(feature_config, label_maps).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    best_state = _cpu_state_dict(model)
    best_epoch = 0
    best_score = float("inf")
    non_improving_epochs = 0
    stopped_epoch = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in _batched(split.train, batch_size):
            optimizer.zero_grad()
            (
                token_ids,
                offsets,
                dataset_variant_ids,
                model_key_ids,
                variant_ids,
                tool_ids,
                numeric_features,
                labels,
            ) = _gate_batch(
                batch,
                feature_config=feature_config,
                label_maps=label_maps,
                device=device,
            )
            logits = model(
                token_ids,
                offsets,
                dataset_variant_ids,
                model_key_ids,
                variant_ids,
                tool_ids,
                numeric_features,
            )
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            total_count += len(batch)
        train_loss = (total_loss / total_count) if total_count else 0.0
        val_loss = gate_loss(
            model,
            split.val,
            feature_config=feature_config,
            label_maps=label_maps,
            batch_size=batch_size,
            device=device,
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        score = val_loss if split.val else train_loss
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = _cpu_state_dict(model)
            non_improving_epochs = 0
        else:
            non_improving_epochs += 1
            if (
                early_stopping_patience > 0
                and split.val
                and non_improving_epochs >= early_stopping_patience
            ):
                stopped_epoch = epoch + 1
                _log_epoch(
                    component="tool_gate",
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_epoch=best_epoch,
                )
                break
        _log_epoch(
            component="tool_gate",
            epoch=epoch + 1,
            total_epochs=epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            best_epoch=best_epoch,
        )

    model.load_state_dict(best_state)

    metrics = evaluate_gate(
        model,
        split,
        feature_config=feature_config,
        label_maps=label_maps,
        batch_size=batch_size,
        device=device,
    )
    metrics["history"] = history
    metrics["best_epoch"] = float(best_epoch)
    metrics["stopped_epoch"] = float(stopped_epoch)
    metrics["early_stopped"] = bool(stopped_epoch)
    print(
        "Finished tool gate: "
        f"train_acc={metrics['train']['accuracy']:.4f} "
        f"val_acc={metrics['val']['accuracy']:.4f} "
        f"train_loss={metrics['train']['loss']:.6f} "
        f"val_loss={metrics['val']['loss']:.6f} "
        f"best_epoch={metrics['best_epoch']:.0f}",
        flush=True,
    )
    return model, metrics


def evaluate_selector(
    model: nn.Module,
    split: TrainEvalSplit,
    *,
    feature_config: FeatureConfig,
    label_maps: LabelMaps,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    def evaluate_subset(subset: Sequence[dict[str, Any]]) -> dict[str, float]:
        if not subset:
            return {"count": 0.0, "accuracy": 0.0, "loss": 0.0}
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in _batched(subset, batch_size):
                (
                    token_ids,
                    offsets,
                    dataset_variant_ids,
                    model_key_ids,
                    numeric_features,
                    labels,
                ) = _selector_batch(
                    batch,
                    feature_config=feature_config,
                    label_maps=label_maps,
                    device=device,
                )
                logits = model(
                    token_ids,
                    offsets,
                    dataset_variant_ids,
                    model_key_ids,
                    numeric_features,
                )
                predictions = torch.argmax(logits, dim=1)
                correct += int((predictions == labels).sum().item())
                total += len(batch)
        return {
            "count": float(total),
            "accuracy": float(correct / total) if total else 0.0,
            "loss": selector_loss(
                model,
                subset,
                feature_config=feature_config,
                label_maps=label_maps,
                batch_size=batch_size,
                device=device,
            ),
        }

    return {
        "train": evaluate_subset(split.train),
        "val": evaluate_subset(split.val),
    }


def evaluate_gate(
    model: nn.Module,
    split: TrainEvalSplit,
    *,
    feature_config: FeatureConfig,
    label_maps: LabelMaps,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    def evaluate_subset(subset: Sequence[dict[str, Any]]) -> dict[str, float]:
        if not subset:
            return {
                "count": 0.0,
                "accuracy": 0.0,
                "loss": 0.0,
                "positive_rate": 0.0,
            }
        model.eval()
        correct = 0
        total = 0
        predicted_positive = 0
        with torch.no_grad():
            for batch in _batched(subset, batch_size):
                (
                    token_ids,
                    offsets,
                    dataset_variant_ids,
                    model_key_ids,
                    variant_ids,
                    tool_ids,
                    numeric_features,
                    labels,
                ) = _gate_batch(
                    batch,
                    feature_config=feature_config,
                    label_maps=label_maps,
                    device=device,
                )
                logits = model(
                    token_ids,
                    offsets,
                    dataset_variant_ids,
                    model_key_ids,
                    variant_ids,
                    tool_ids,
                    numeric_features,
                )
                predictions = (torch.sigmoid(logits) >= 0.5).to(labels.dtype)
                correct += int((predictions == labels).sum().item())
                predicted_positive += int(predictions.sum().item())
                total += len(batch)
        return {
            "count": float(total),
            "accuracy": float(correct / total) if total else 0.0,
            "loss": gate_loss(
                model,
                subset,
                feature_config=feature_config,
                label_maps=label_maps,
                batch_size=batch_size,
                device=device,
            ),
            "positive_rate": float(predicted_positive / total) if total else 0.0,
        }

    return {
        "train": evaluate_subset(split.train),
        "val": evaluate_subset(split.val),
    }


def _training_summary_markdown(
    *,
    row_count: int,
    selector_examples: Sequence[dict[str, Any]],
    gate_examples: Sequence[dict[str, Any]],
    feature_config: FeatureConfig,
    selector_metrics: dict[str, Any],
    gate_metrics: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# GateKG Training Summary",
            "",
            f"- Dataset rows: {row_count}",
            f"- Selector examples: {len(selector_examples)}",
            f"- Tool-gate examples: {len(gate_examples)}",
            f"- Health metrics: {', '.join(feature_config.health_metric_names) or 'none'}",
            f"- Selector train accuracy: {selector_metrics['train']['accuracy']:.4f}",
            f"- Selector val accuracy: {selector_metrics['val']['accuracy']:.4f}",
            f"- Selector train loss: {selector_metrics['train']['loss']:.4f}",
            f"- Selector val loss: {selector_metrics['val']['loss']:.4f}",
            f"- Selector best epoch: {selector_metrics['best_epoch']:.0f}",
            f"- Selector generalization gap: {selector_metrics['generalization_gap']:.4f}",
            f"- Gate train accuracy: {gate_metrics['train']['accuracy']:.4f}",
            f"- Gate val accuracy: {gate_metrics['val']['accuracy']:.4f}",
            f"- Gate train loss: {gate_metrics['train']['loss']:.4f}",
            f"- Gate val loss: {gate_metrics['val']['loss']:.4f}",
            f"- Gate best epoch: {gate_metrics['best_epoch']:.0f}",
        ]
    )


def _training_results_payload(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    selector = metrics_payload["selector"]
    tool_gate = metrics_payload["tool_gate"]
    return {
        "dataset": dict(metrics_payload["dataset"]),
        "training": dict(metrics_payload["training"]),
        "selector": {
            "train_accuracy": selector["train"]["accuracy"],
            "val_accuracy": selector["val"]["accuracy"],
            "train_loss": selector["train"]["loss"],
            "val_loss": selector["val"]["loss"],
            "best_epoch": selector["best_epoch"],
            "early_stopped": selector["early_stopped"],
            "generalization_gap": selector["generalization_gap"],
        },
        "tool_gate": {
            "train_accuracy": tool_gate["train"]["accuracy"],
            "val_accuracy": tool_gate["val"]["accuracy"],
            "train_loss": tool_gate["train"]["loss"],
            "val_loss": tool_gate["val"]["loss"],
            "train_positive_rate": tool_gate["train"]["positive_rate"],
            "val_positive_rate": tool_gate["val"]["positive_rate"],
            "best_epoch": tool_gate["best_epoch"],
            "early_stopped": tool_gate["early_stopped"],
        },
    }


def _write_training_log(
    path: Path,
    *,
    metrics_payload: dict[str, Any],
    results_payload: dict[str, Any],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "config",
                    "dataset": metrics_payload["dataset"],
                    "training": metrics_payload["training"],
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n"
        )
        for component_name, metrics_key in (
            ("selector", "selector"),
            ("tool_gate", "tool_gate"),
        ):
            for epoch_payload in metrics_payload[metrics_key]["history"]:
                handle.write(
                    json.dumps(
                        {
                            "event": "epoch",
                            "component": component_name,
                            **epoch_payload,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    + "\n"
                )
        handle.write(
            json.dumps(
                {
                    "event": "final",
                    "results": results_payload,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n"
        )


def train_gatekg(
    *,
    dataset_path: str | Path,
    artifact_dir: str | Path,
    selector_epochs: int,
    gate_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    group_split: bool = True,
    include_aggressive_write: bool = False,
    selector_dropout: float = DEFAULT_SELECTOR_DROPOUT,
    gate_dropout: float = DEFAULT_GATE_DROPOUT,
    gate_label_source: str = "reward_divergence",
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = GATEKG_VARIANTS if include_aggressive_write else GATEKG_VARIANTS[:2]
    rows = [
        row
        for row in read_dataset_rows(dataset_path)
        if normalize_variant_name(row["variant"]) in variants
    ]
    if not rows:
        raise ValueError("Training dataset is empty for the selected routing modes.")

    if gate_label_source == "reward_divergence":
        gate_builder = build_gate_examples
    elif gate_label_source == "trajectory_state":
        gate_builder = build_trajectory_gate_examples
    else:
        raise ValueError(f"Unsupported gate label source: {gate_label_source}")

    if group_split:
        # Partition trajectories before computing cohort rewards or selecting targets.
        row_split = split_examples(rows, seed=seed)
        selector_split = TrainEvalSplit(
            build_selector_targets(row_split.train), build_selector_targets(row_split.val)
        )
        gate_split = TrainEvalSplit(gate_builder(row_split.train), gate_builder(row_split.val))
    else:
        selector_split = split_examples(
            build_selector_targets(rows), seed=seed, group_by_identity=False
        )
        gate_split = split_examples(gate_builder(rows), seed=seed + 1, group_by_identity=False)
    selector_examples = selector_split.train + selector_split.val
    gate_examples = gate_split.train + gate_split.val
    if not gate_split.train:
        raise ValueError("Tool-gate training examples are empty.")

    metric_names = sorted(
        {
            metric_name
            for row in selector_split.train + gate_split.train
            for metric_name in dict(row.get("baseline_metric_scores") or {}).keys()
        }
    )
    feature_config = build_feature_config(
        metric_names,
        selector_dropout=selector_dropout,
        gate_dropout=gate_dropout,
    )
    label_maps = build_label_maps(
        rows=selector_split.train, gate_examples=gate_split.train, variants=variants
    )

    selector_model, selector_metrics = train_selector(
        selector_split,
        feature_config=feature_config,
        label_maps=label_maps,
        epochs=selector_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        device=device,
    )
    gate_model, gate_metrics = train_gate(
        gate_split,
        feature_config=feature_config,
        label_maps=label_maps,
        epochs=gate_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        device=device,
    )

    resolved_artifact_dir = Path(artifact_dir)
    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        component: {
            fold: sorted(
                {
                    (str(row["dataset_variant"]), str(row["model_key"]), str(row["entry_id"]))
                    for row in examples
                }
            )
            for fold, examples in (("train", split.train), ("val", split.val))
        }
        for component, split in (("selector", selector_split), ("gate", gate_split))
    }
    (resolved_artifact_dir / "split.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    save_artifact_metadata(
        resolved_artifact_dir,
        feature_config=feature_config,
        label_maps=label_maps,
    )
    torch.save(
        selector_model.cpu().state_dict(),
        resolved_artifact_dir / "selector.pt",
    )
    torch.save(
        gate_model.cpu().state_dict(),
        resolved_artifact_dir / "tool_gate.pt",
    )

    metrics_payload = {
        "dataset": {
            "path": str(dataset_path),
            "row_count": len(rows),
            "selector_example_count": len(selector_examples),
            "gate_example_count": len(gate_examples),
        },
        "training": {
            "seed": seed,
            "device": str(device),
            "selector_epochs": selector_epochs,
            "gate_epochs": gate_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "group_split": group_split,
            "selector_dropout": selector_dropout,
            "gate_dropout": gate_dropout,
            "gate_label_source": gate_label_source,
        },
        "selector": selector_metrics,
        "tool_gate": gate_metrics,
    }
    results_payload = _training_results_payload(metrics_payload)
    (resolved_artifact_dir / "training_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (resolved_artifact_dir / "training_results.json").write_text(
        json.dumps(results_payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_training_log(
        resolved_artifact_dir / "training_log.jsonl",
        metrics_payload=metrics_payload,
        results_payload=results_payload,
    )
    (resolved_artifact_dir / "training_summary.md").write_text(
        _training_summary_markdown(
            row_count=len(rows),
            selector_examples=selector_examples,
            gate_examples=gate_examples,
            feature_config=feature_config,
            selector_metrics=selector_metrics,
            gate_metrics=gate_metrics,
        ),
        encoding="utf-8",
    )
    return metrics_payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = train_gatekg(
        dataset_path=args.dataset,
        artifact_dir=args.artifact_dir,
        selector_epochs=args.selector_epochs,
        gate_epochs=args.gate_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        group_split=args.group_split,
        include_aggressive_write=args.include_aggressive_write,
        selector_dropout=args.selector_dropout,
        gate_dropout=args.gate_dropout,
        gate_label_source=args.gate_label_source,
    )
    print(
        "Trained GateKG artifacts. "
        f"selector_examples={metrics['dataset']['selector_example_count']} "
        f"gate_examples={metrics['dataset']['gate_example_count']}"
    )
    print(
        "Selector: "
        f"train_acc={metrics['selector']['train']['accuracy']:.4f} "
        f"val_acc={metrics['selector']['val']['accuracy']:.4f} "
        f"train_loss={metrics['selector']['train']['loss']:.4f} "
        f"val_loss={metrics['selector']['val']['loss']:.4f}"
    )
    print(
        "Tool gate: "
        f"train_acc={metrics['tool_gate']['train']['accuracy']:.4f} "
        f"val_acc={metrics['tool_gate']['val']['accuracy']:.4f} "
        f"train_loss={metrics['tool_gate']['train']['loss']:.4f} "
        f"val_loss={metrics['tool_gate']['val']['loss']:.4f}"
    )
    print(f"Wrote artifacts to {args.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
