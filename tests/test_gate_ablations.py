from types import SimpleNamespace

import torch

from gatekg.training.features import build_feature_config, build_gate_model, build_label_maps
from scripts.compare_gates import encode


def test_ablation_masks_exact_classifier_input_coordinates():
    example = dict(
        question="Which city?",
        variant="read_write_rlm",
        tool_name="search_nodes",
        label=1,
        baseline_average_health=0.7,
        baseline_metric_scores={"coverage": 0.9},
        call_index=3,
        prior_call_count=3,
        prior_mutating_call_count=1,
        prior_same_tool_count=1,
        prior_unique_tool_count=2,
        writable_flag=1,
    )
    config = build_feature_config(["coverage"])
    maps = build_label_maps(rows=[example], gate_examples=[example])
    model = build_gate_model(config, maps).eval()
    inputs = []
    model.head.register_forward_pre_hook(lambda _, args: inputs.append(args[0].clone()))
    bundle = SimpleNamespace(feature_config=config, label_maps=maps, tool_gate_model=model)
    encode(bundle, [example])
    full, zero_health, zero_trajectory, zero_question = inputs
    numeric_start = config.text_embed_dim + 2 * config.category_embed_dim
    for masked, coordinates in [
        (zero_health, slice(numeric_start, numeric_start + 2)),
        (zero_trajectory, slice(numeric_start + 2, None)),
        (zero_question, slice(0, config.text_embed_dim)),
    ]:
        expected = full.clone()
        assert expected[:, coordinates].count_nonzero() > 0
        expected[:, coordinates] = 0
        assert torch.equal(expected, masked)
