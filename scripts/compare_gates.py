"""Evaluate fitted-gate ablations and classifier baselines on its question split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gatekg.training.dataset import (
    build_gate_examples,
    build_trajectory_gate_examples,
    read_dataset_rows,
)
from gatekg.training.features import load_inference_bundle, normalize_variant_name, tool_is_mutating
from gatekg.training.train import _gate_batch, split_examples


def metrics(target, scores, threshold=0.5):
    return {
        "accuracy": float(accuracy_score(target, scores >= threshold)),
        "auc": float(roc_auc_score(target, scores)) if len(set(target)) == 2 else None,
    }


@torch.inference_mode()
def encode(bundle, examples):
    config = bundle.feature_config
    gate = bundle.tool_gate_model
    tokens, offsets, _, _, variant, tool, numeric, labels = _gate_batch(
        examples, feature_config=config, label_maps=bundle.label_maps, device=torch.device("cpu")
    )
    text = gate.question_embed(tokens, offsets)
    categories = torch.cat((gate.variant_embed(variant), gate.tool_embed(tool)), dim=1)
    health_size = 1 + len(config.health_metric_names)
    normalized = gate.numeric_norm(numeric)
    ablations = {}
    for mode in ("full", "zero_health", "zero_trajectory", "zero_question"):
        numbers = normalized.clone()
        if mode == "zero_health":
            numbers[:, :health_size] = 0
        if mode == "zero_trajectory":
            numbers[:, health_size:] = 0
        question = torch.zeros_like(text) if mode == "zero_question" else text
        features = torch.cat((question, categories, numbers), dim=1)
        ablations[mode] = torch.sigmoid(gate.head(features).flatten()).numpy()
    structure = torch.cat((categories, normalized), dim=1)
    return text.numpy(), structure.numpy(), labels.numpy(), ablations


def compare(dataset, artifact_dir, semantic):
    bundle = load_inference_bundle(artifact_dir)
    training = json.loads((artifact_dir / "training_metrics.json").read_text())["training"]
    if not training["group_split"]:
        raise ValueError("Use a policy trained with a grouped split")
    rows = [
        r
        for r in read_dataset_rows(dataset)
        if normalize_variant_name(r["variant"]) in bundle.label_maps.variant
    ]
    split = split_examples(rows, seed=training["seed"])
    build = (
        build_trajectory_gate_examples
        if training["gate_label_source"] == "trajectory_state"
        else build_gate_examples
    )
    train, val = build(split.train), build(split.val)
    saved = json.loads((artifact_dir / "split.json").read_text())["gate"]
    for fold, examples in (("train", train), ("val", val)):
        keys = {
            (str(e["dataset_variant"]), str(e["model_key"]), str(e["entry_id"])) for e in examples
        }
        if not keys or keys != {tuple(key) for key in saved[fold]}:
            raise ValueError(f"Dataset does not match the nonempty {fold} partition")
    q_train, struct_train, y_train, _ = encode(bundle, train)
    q_val, struct_val, y_val, ablations = encode(bundle, val)
    result = {
        "train_calls": len(train),
        "validation_calls": len(val),
        "fitted_gate": {
            mode: metrics(y_val, scores, bundle.feature_config.gate_threshold)
            for mode, scores in ablations.items()
        },
    }
    if all("duplicate_ratio" in e["baseline_metric_raw_values"] for e in val):
        allowed = []
        for e in val:
            step = e["call_index"] + 1
            write = tool_is_mutating(e["tool_name"])
            blocked = (
                (step > 6 or e["baseline_metric_raw_values"]["duplicate_ratio"] > 0.1)
                if write
                else step > 8
            )
            allowed.append(int(not blocked))
        result["rule_accuracy"] = float(accuracy_score(y_val, allowed))
    encodings = {"hash": (q_train, q_val)}
    if semantic:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            device="cpu",
        )
        questions = list(dict.fromkeys(e["question"] for e in train + val))
        vectors = dict(zip(questions, encoder.encode(questions), strict=True))
        encodings["sbert"] = (
            np.stack([vectors[e["question"]] for e in train]),
            np.stack([vectors[e["question"]] for e in val]),
        )
    for name, (text_train, text_val) in encodings.items():
        x_train = np.column_stack((text_train, struct_train))
        x_val = np.column_stack((text_val, struct_val))
        for classifier, estimator in (
            ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
            ("forest", RandomForestClassifier(n_estimators=100, random_state=training["seed"])),
            (
                "mlp",
                make_pipeline(
                    StandardScaler(),
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64), max_iter=200, random_state=training["seed"]
                    ),
                ),
            ),
        ):
            estimator.fit(x_train, y_train)
            result[f"{name}_{classifier}"] = metrics(y_val, estimator.predict_proba(x_val)[:, 1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument(
        "--semantic", action="store_true", help="Also evaluate the pinned 384-D sentence encoder"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            compare(args.dataset, args.artifact_dir, args.semantic), indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
