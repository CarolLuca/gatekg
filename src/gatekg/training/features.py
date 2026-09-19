"""Policy features and model definitions."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gatekg.paths import policy_artifact_dir

GATEKG_VARIANTS = (
    "read_only_rlm",
    "read_write_rlm",
    "aggressive_write_rlm",
)
MUTATING_TOOL_NAMES = {
    "add_episode_bulk",
    "add_memory",
    "add_triplet",
    "build_communities",
    "build_indices_and_constraints",
    "clear_graph",
    "close",
    "delete_entity_edge",
    "delete_entity_edges",
    "delete_episode",
    "delete_episodes",
    "init_knowledge",
    "initialize",
}
UNKNOWN_LABEL = "__unknown__"
DEFAULT_TOKEN_HASH_BUCKETS = 8192
DEFAULT_SELECTOR_HIDDEN_DIM = 128
DEFAULT_GATE_HIDDEN_DIM = 128
DEFAULT_TEXT_EMBED_DIM = 64
DEFAULT_CATEGORY_EMBED_DIM = 16
DEFAULT_SELECTOR_DROPOUT = 0.0
DEFAULT_GATE_DROPOUT = 0.0
DEFAULT_SELECTOR_THRESHOLD = 0.0
DEFAULT_GATE_THRESHOLD = 0.5
TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def normalize_variant_name(value: str) -> str:
    for variant in GATEKG_VARIANTS:
        if value == variant or value.startswith(f"{variant}_"):
            return variant
    raise ValueError(f"Unsupported GateKG variant value: {value}")


def writable_variant(variant: str) -> bool:
    return normalize_variant_name(variant) != "read_only_rlm"


def tool_is_mutating(tool_name: str) -> bool:
    return tool_name in MUTATING_TOOL_NAMES


def tokenize_question(question: str) -> list[str]:
    return TOKEN_RE.findall(question.lower())


def _stable_hash(text: str) -> int:
    return int(blake2b(text.encode("utf-8"), digest_size=8).hexdigest(), 16)


def hash_token(token: str, *, buckets: int) -> int:
    if buckets <= 0:
        raise ValueError("Token hash bucket count must be positive.")
    return _stable_hash(token) % buckets


def encode_question(question: str, *, buckets: int) -> list[int]:
    tokens = tokenize_question(question)
    if not tokens:
        return [hash_token("__empty__", buckets=buckets)]
    return [hash_token(token, buckets=buckets) for token in tokens]


def tensorize_question_batch(
    questions: Sequence[str],
    *,
    buckets: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_token_ids: list[int] = []
    offsets: list[int] = []
    cursor = 0
    for question in questions:
        offsets.append(cursor)
        encoded = encode_question(question, buckets=buckets)
        flat_token_ids.extend(encoded)
        cursor += len(encoded)

    return (
        torch.tensor(flat_token_ids, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
    )


def build_input_label_map(values: Iterable[str]) -> dict[str, int]:
    unique_values = sorted({str(value) for value in values if str(value).strip()})
    return {
        UNKNOWN_LABEL: 0,
        **{value: index + 1 for index, value in enumerate(unique_values)},
    }


def encode_input_label(label_map: dict[str, int], value: str | None) -> int:
    if value is None:
        return label_map[UNKNOWN_LABEL]
    return label_map.get(str(value), label_map[UNKNOWN_LABEL])


def build_variant_label_map(variants: Sequence[str] = GATEKG_VARIANTS) -> dict[str, int]:
    return {variant: index for index, variant in enumerate(variants)}


def invert_label_map(label_map: dict[str, int]) -> dict[int, str]:
    return {index: label for label, index in label_map.items()}


@dataclass(frozen=True)
class LabelMaps:
    dataset_variant: dict[str, int]
    model_key: dict[str, int]
    variant: dict[str, int]
    tool_name: dict[str, int]

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            "dataset_variant": dict(self.dataset_variant),
            "model_key": dict(self.model_key),
            "variant": dict(self.variant),
            "tool_name": dict(self.tool_name),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LabelMaps:
        return cls(
            dataset_variant={
                str(label): int(index) for label, index in dict(payload["dataset_variant"]).items()
            },
            model_key={
                str(label): int(index) for label, index in dict(payload["model_key"]).items()
            },
            variant={str(label): int(index) for label, index in dict(payload["variant"]).items()},
            tool_name={
                str(label): int(index) for label, index in dict(payload["tool_name"]).items()
            },
        )


def selector_numeric_feature_names(health_metric_names: Sequence[str]) -> list[str]:
    return [
        "baseline_average_health",
        *(f"baseline_metric::{metric_name}" for metric_name in health_metric_names),
        "context_chunk_count",
        "kg_context_chunk_count",
    ]


def gate_numeric_feature_names(health_metric_names: Sequence[str]) -> list[str]:
    return [
        "baseline_average_health",
        *(f"baseline_metric::{metric_name}" for metric_name in health_metric_names),
        "call_index",
        "prior_call_count",
        "prior_mutating_call_count",
        "prior_same_tool_count",
        "prior_unique_tool_count",
        "writable_flag",
    ]


@dataclass(frozen=True)
class FeatureConfig:
    token_hash_buckets: int
    health_metric_names: list[str]
    selector_numeric_feature_names: list[str]
    gate_numeric_feature_names: list[str]
    selector_hidden_dim: int = DEFAULT_SELECTOR_HIDDEN_DIM
    gate_hidden_dim: int = DEFAULT_GATE_HIDDEN_DIM
    text_embed_dim: int = DEFAULT_TEXT_EMBED_DIM
    category_embed_dim: int = DEFAULT_CATEGORY_EMBED_DIM
    selector_dropout: float = DEFAULT_SELECTOR_DROPOUT
    gate_dropout: float = DEFAULT_GATE_DROPOUT
    selector_threshold: float = DEFAULT_SELECTOR_THRESHOLD
    gate_threshold: float = DEFAULT_GATE_THRESHOLD
    version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "token_hash_buckets": self.token_hash_buckets,
            "health_metric_names": list(self.health_metric_names),
            "selector_numeric_feature_names": list(self.selector_numeric_feature_names),
            "gate_numeric_feature_names": list(self.gate_numeric_feature_names),
            "selector_hidden_dim": self.selector_hidden_dim,
            "gate_hidden_dim": self.gate_hidden_dim,
            "text_embed_dim": self.text_embed_dim,
            "category_embed_dim": self.category_embed_dim,
            "selector_dropout": self.selector_dropout,
            "gate_dropout": self.gate_dropout,
            "selector_threshold": self.selector_threshold,
            "gate_threshold": self.gate_threshold,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureConfig:
        return cls(
            version=int(payload.get("version", 1)),
            token_hash_buckets=int(payload["token_hash_buckets"]),
            health_metric_names=list(payload["health_metric_names"]),
            selector_numeric_feature_names=list(payload["selector_numeric_feature_names"]),
            gate_numeric_feature_names=list(payload["gate_numeric_feature_names"]),
            selector_hidden_dim=int(
                payload.get("selector_hidden_dim", DEFAULT_SELECTOR_HIDDEN_DIM)
            ),
            gate_hidden_dim=int(payload.get("gate_hidden_dim", DEFAULT_GATE_HIDDEN_DIM)),
            text_embed_dim=int(payload.get("text_embed_dim", DEFAULT_TEXT_EMBED_DIM)),
            category_embed_dim=int(payload.get("category_embed_dim", DEFAULT_CATEGORY_EMBED_DIM)),
            selector_dropout=float(payload.get("selector_dropout", DEFAULT_SELECTOR_DROPOUT)),
            gate_dropout=float(payload.get("gate_dropout", DEFAULT_GATE_DROPOUT)),
            selector_threshold=float(payload.get("selector_threshold", DEFAULT_SELECTOR_THRESHOLD)),
            gate_threshold=float(payload.get("gate_threshold", DEFAULT_GATE_THRESHOLD)),
        )


def build_feature_config(
    metric_names: Iterable[str],
    *,
    selector_dropout: float = DEFAULT_SELECTOR_DROPOUT,
    gate_dropout: float = DEFAULT_GATE_DROPOUT,
) -> FeatureConfig:
    resolved_metric_names = sorted(
        {str(metric_name) for metric_name in metric_names if metric_name}
    )
    return FeatureConfig(
        token_hash_buckets=DEFAULT_TOKEN_HASH_BUCKETS,
        health_metric_names=resolved_metric_names,
        selector_numeric_feature_names=selector_numeric_feature_names(resolved_metric_names),
        gate_numeric_feature_names=gate_numeric_feature_names(resolved_metric_names),
        selector_dropout=float(selector_dropout),
        gate_dropout=float(gate_dropout),
    )


def build_label_maps(
    *,
    rows: Sequence[dict[str, Any]],
    gate_examples: Sequence[dict[str, Any]],
    variants: Sequence[str] = GATEKG_VARIANTS,
) -> LabelMaps:
    return LabelMaps(
        dataset_variant=build_input_label_map(
            str(row.get("dataset_variant") or "") for row in rows
        ),
        model_key=build_input_label_map(str(row.get("model_key") or "") for row in rows),
        variant=build_variant_label_map(variants),
        tool_name=build_input_label_map(
            str(example.get("tool_name") or "") for example in gate_examples
        ),
    )


def _metric_score_map(row: dict[str, Any], key: str) -> dict[str, float]:
    payload = row.get(key) or {}
    return {
        str(metric_name): float(metric_value) for metric_name, metric_value in dict(payload).items()
    }


def selector_numeric_vector(
    row: dict[str, Any],
    *,
    health_metric_names: Sequence[str],
) -> list[float]:
    metric_scores = _metric_score_map(row, "baseline_metric_scores")
    return [
        float(row.get("baseline_average_health") or 0.0),
        *(metric_scores.get(metric_name, 0.0) for metric_name in health_metric_names),
        float(row.get("context_chunk_count") or 0.0),
        float(row.get("kg_context_chunk_count") or 0.0),
    ]


def gate_numeric_vector(
    row: dict[str, Any],
    *,
    health_metric_names: Sequence[str],
    call_index: int,
    prior_call_count: int,
    prior_mutating_call_count: int,
    prior_same_tool_count: int,
    prior_unique_tool_count: int,
    writable_flag: int,
) -> list[float]:
    metric_scores = _metric_score_map(row, "baseline_metric_scores")
    return [
        float(row.get("baseline_average_health") or 0.0),
        *(metric_scores.get(metric_name, 0.0) for metric_name in health_metric_names),
        float(call_index),
        float(prior_call_count),
        float(prior_mutating_call_count),
        float(prior_same_tool_count),
        float(prior_unique_tool_count),
        float(writable_flag),
    ]


def as_float_tensor(
    values: Sequence[Sequence[float]],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


class SelectorModel(nn.Module):
    def __init__(
        self,
        *,
        token_hash_buckets: int,
        dataset_variant_count: int,
        model_key_count: int,
        numeric_feature_dim: int,
        text_embed_dim: int,
        category_embed_dim: int,
        hidden_dim: int,
        class_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.question_embed = nn.EmbeddingBag(
            token_hash_buckets,
            text_embed_dim,
            mode="mean",
        )
        self.numeric_norm = nn.LayerNorm(numeric_feature_dim)
        self.head = nn.Sequential(
            nn.Linear(
                text_embed_dim + numeric_feature_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, class_count),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        offsets: torch.Tensor,
        dataset_variant_ids: torch.Tensor,
        model_key_ids: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        question_features = self.question_embed(token_ids, offsets)
        normalized_numeric = self.numeric_norm(numeric_features)
        return self.head(
            torch.cat(
                [question_features, normalized_numeric],
                dim=1,
            )
        )


class ToolGateModel(nn.Module):
    def __init__(
        self,
        *,
        token_hash_buckets: int,
        dataset_variant_count: int,
        model_key_count: int,
        variant_count: int,
        tool_count: int,
        numeric_feature_dim: int,
        text_embed_dim: int,
        category_embed_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.question_embed = nn.EmbeddingBag(
            token_hash_buckets,
            text_embed_dim,
            mode="mean",
        )
        self.variant_embed = nn.Embedding(variant_count, category_embed_dim)
        self.tool_embed = nn.Embedding(tool_count, category_embed_dim)
        self.numeric_norm = nn.LayerNorm(numeric_feature_dim)
        self.head = nn.Sequential(
            nn.Linear(
                text_embed_dim + (2 * category_embed_dim) + numeric_feature_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        offsets: torch.Tensor,
        dataset_variant_ids: torch.Tensor,
        model_key_ids: torch.Tensor,
        variant_ids: torch.Tensor,
        tool_ids: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        question_features = self.question_embed(token_ids, offsets)
        categorical_features = torch.cat(
            [
                self.variant_embed(variant_ids),
                self.tool_embed(tool_ids),
            ],
            dim=1,
        )
        normalized_numeric = self.numeric_norm(numeric_features)
        return self.head(
            torch.cat(
                [question_features, categorical_features, normalized_numeric],
                dim=1,
            )
        ).squeeze(1)


def build_selector_model(
    feature_config: FeatureConfig,
    label_maps: LabelMaps,
) -> SelectorModel:
    return SelectorModel(
        token_hash_buckets=feature_config.token_hash_buckets,
        dataset_variant_count=len(label_maps.dataset_variant),
        model_key_count=len(label_maps.model_key),
        numeric_feature_dim=len(feature_config.selector_numeric_feature_names),
        text_embed_dim=feature_config.text_embed_dim,
        category_embed_dim=feature_config.category_embed_dim,
        hidden_dim=feature_config.selector_hidden_dim,
        class_count=len(label_maps.variant),
        dropout=feature_config.selector_dropout,
    )


def build_gate_model(
    feature_config: FeatureConfig,
    label_maps: LabelMaps,
) -> ToolGateModel:
    return ToolGateModel(
        token_hash_buckets=feature_config.token_hash_buckets,
        dataset_variant_count=len(label_maps.dataset_variant),
        model_key_count=len(label_maps.model_key),
        variant_count=len(label_maps.variant),
        tool_count=len(label_maps.tool_name),
        numeric_feature_dim=len(feature_config.gate_numeric_feature_names),
        text_embed_dim=feature_config.text_embed_dim,
        category_embed_dim=feature_config.category_embed_dim,
        hidden_dim=feature_config.gate_hidden_dim,
        dropout=feature_config.gate_dropout,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_artifact_metadata(
    artifact_dir: str | Path,
    *,
    feature_config: FeatureConfig,
    label_maps: LabelMaps,
) -> None:
    resolved_artifact_dir = policy_artifact_dir(artifact_dir)
    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(resolved_artifact_dir / "feature_config.json", feature_config.to_dict())
    _write_json(resolved_artifact_dir / "label_maps.json", label_maps.to_dict())


@dataclass(frozen=True)
class InferenceBundle:
    feature_config: FeatureConfig
    label_maps: LabelMaps
    selector_model: SelectorModel
    tool_gate_model: ToolGateModel


def load_inference_bundle(
    artifact_dir: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
) -> InferenceBundle:
    resolved_artifact_dir = policy_artifact_dir(artifact_dir)
    resolved_device = torch.device(device or "cpu")
    feature_config = FeatureConfig.from_dict(
        json.loads((resolved_artifact_dir / "feature_config.json").read_text(encoding="utf-8"))
    )
    label_maps = LabelMaps.from_dict(
        json.loads((resolved_artifact_dir / "label_maps.json").read_text(encoding="utf-8"))
    )
    selector_model = build_selector_model(feature_config, label_maps).to(resolved_device)
    selector_model.load_state_dict(
        torch.load(
            resolved_artifact_dir / "selector.pt", map_location=resolved_device, weights_only=True
        )
    )
    selector_model.eval()

    tool_gate_model = build_gate_model(feature_config, label_maps).to(resolved_device)
    tool_gate_model.load_state_dict(
        torch.load(resolved_artifact_dir / "tool_gate.pt", map_location=resolved_device)
    )
    tool_gate_model.eval()

    return InferenceBundle(
        feature_config=feature_config,
        label_maps=label_maps,
        selector_model=selector_model,
        tool_gate_model=tool_gate_model,
    )


def selector_scores_from_row(
    *,
    model: SelectorModel,
    feature_config: FeatureConfig,
    label_maps: LabelMaps,
    row: dict[str, Any],
    device: torch.device | None = None,
) -> dict[str, float]:
    resolved_device = device or next(model.parameters()).device
    token_ids, offsets = tensorize_question_batch(
        [str(row.get("question") or "")],
        buckets=feature_config.token_hash_buckets,
        device=resolved_device,
    )
    numeric_features = as_float_tensor(
        [
            selector_numeric_vector(
                row,
                health_metric_names=feature_config.health_metric_names,
            )
        ],
        device=resolved_device,
    )
    dataset_variant_ids = torch.tensor(
        [
            encode_input_label(
                label_maps.dataset_variant,
                str(row.get("dataset_variant") or ""),
            )
        ],
        dtype=torch.long,
        device=resolved_device,
    )
    model_key_ids = torch.tensor(
        [
            encode_input_label(
                label_maps.model_key,
                str(row.get("model_key") or ""),
            )
        ],
        dtype=torch.long,
        device=resolved_device,
    )
    with torch.no_grad():
        logits = model(
            token_ids,
            offsets,
            dataset_variant_ids,
            model_key_ids,
            numeric_features,
        )
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    inverse_variant_map = invert_label_map(label_maps.variant)
    return {
        inverse_variant_map[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }


def gate_score_from_row(
    *,
    model: ToolGateModel,
    feature_config: FeatureConfig,
    label_maps: LabelMaps,
    row: dict[str, Any],
    tool_name: str,
    call_index: int,
    prior_call_count: int,
    prior_mutating_call_count: int,
    prior_same_tool_count: int,
    prior_unique_tool_count: int,
    writable_flag: int,
    device: torch.device | None = None,
) -> float:
    resolved_device = device or next(model.parameters()).device
    token_ids, offsets = tensorize_question_batch(
        [str(row.get("question") or "")],
        buckets=feature_config.token_hash_buckets,
        device=resolved_device,
    )
    numeric_features = as_float_tensor(
        [
            gate_numeric_vector(
                row,
                health_metric_names=feature_config.health_metric_names,
                call_index=call_index,
                prior_call_count=prior_call_count,
                prior_mutating_call_count=prior_mutating_call_count,
                prior_same_tool_count=prior_same_tool_count,
                prior_unique_tool_count=prior_unique_tool_count,
                writable_flag=writable_flag,
            )
        ],
        device=resolved_device,
    )
    dataset_variant_ids = torch.tensor(
        [
            encode_input_label(
                label_maps.dataset_variant,
                str(row.get("dataset_variant") or ""),
            )
        ],
        dtype=torch.long,
        device=resolved_device,
    )
    model_key_ids = torch.tensor(
        [
            encode_input_label(
                label_maps.model_key,
                str(row.get("model_key") or ""),
            )
        ],
        dtype=torch.long,
        device=resolved_device,
    )
    variant_ids = torch.tensor(
        [label_maps.variant[normalize_variant_name(str(row.get("variant") or ""))]],
        dtype=torch.long,
        device=resolved_device,
    )
    tool_ids = torch.tensor(
        [encode_input_label(label_maps.tool_name, tool_name)],
        dtype=torch.long,
        device=resolved_device,
    )
    with torch.no_grad():
        logit = model(
            token_ids,
            offsets,
            dataset_variant_ids,
            model_key_ids,
            variant_ids,
            tool_ids,
            numeric_features,
        )[0]
        probability = torch.sigmoid(logit).item()
    return float(probability)
