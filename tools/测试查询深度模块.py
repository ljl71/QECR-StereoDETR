"""CPU forward/backward checks for all query-depth modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "lib" / "models" / "monodetr" / "query_depth.py"
SPEC = importlib.util.spec_from_file_location("qdr_query_depth", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)
QUERY_DEPTH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUERY_DEPTH
SPEC.loader.exec_module(QUERY_DEPTH)
QueryDepthDistributionRefiner = QUERY_DEPTH.QueryDepthDistributionRefiner
QueryDepthDistributionSampler = QUERY_DEPTH.QueryDepthDistributionSampler
BaselineSafeDepthFusion = QUERY_DEPTH.BaselineSafeDepthFusion
UncertaintyGeometryGate = QUERY_DEPTH.UncertaintyGeometryGate


def assert_probability(tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all()
    expected = torch.ones_like(tensor.sum(dim=-1))
    assert torch.allclose(tensor.sum(dim=-1), expected, atol=1e-5)


def main() -> None:
    torch.manual_seed(7)
    batch_size, num_queries = 2, 6
    hidden_dim, decoder_layers = 32, 3
    valid_bins, height, width = 20, 12, 18

    depth_logits = torch.randn(
        batch_size,
        valid_bins + 1,
        height,
        width,
        requires_grad=True,
    )
    depth_bins = torch.cat(
        [torch.linspace(0.5, 60.0, valid_bins), torch.zeros(1)]
    )
    center_points = torch.rand(batch_size, num_queries, 2)
    box_extents = torch.rand(batch_size, num_queries, 2) * 0.4 + 0.05
    features = [
        torch.randn(
            batch_size, num_queries, hidden_dim, requires_grad=True
        )
        for _ in range(decoder_layers)
    ]

    single_sampler = QueryDepthDistributionSampler(
        hidden_dim, decoder_layers, num_points=1
    )
    single_output = single_sampler(
        depth_logits,
        depth_bins,
        features[0],
        center_points,
        box_extents,
        layer_index=0,
    )
    assert single_output["points"].shape == (
        batch_size,
        num_queries,
        1,
        2,
    )
    assert_probability(single_output["probabilities"])

    multi_sampler = QueryDepthDistributionSampler(
        hidden_dim, decoder_layers, num_points=5
    )
    refiner = QueryDepthDistributionRefiner(
        hidden_dim,
        decoder_layers,
        total_bins=valid_bins,
        local_bins=8,
        enabled=True,
        prior_weight=0.5,
    )
    previous = None
    refined_outputs = []
    for layer_index in range(decoder_layers):
        sampled = multi_sampler(
            depth_logits,
            depth_bins,
            features[layer_index],
            center_points,
            box_extents,
            layer_index=layer_index,
        )
        assert sampled["points"].shape == (
            batch_size,
            num_queries,
            5,
            2,
        )
        assert_probability(sampled["point_weights"].squeeze(-1))
        assert_probability(sampled["probabilities"])
        refined = refiner(
            sampled["probabilities"],
            depth_bins,
            features[layer_index],
            layer_index,
            previous_full_probabilities=previous,
        )
        assert refined["probabilities"].shape == (
            batch_size,
            num_queries,
            8,
        )
        assert_probability(refined["probabilities"])
        assert_probability(refined["full_probabilities"])
        previous = refined["full_probabilities"]
        refined_outputs.append((sampled, refined))

    final_sampled, final_refined = refined_outputs[-1]
    baseline_depth = torch.full(
        (batch_size, num_queries, 1),
        20.0,
        requires_grad=True,
    )
    raw_query_depth = torch.full(
        (batch_size, num_queries, 1),
        30.0,
        requires_grad=True,
    )
    safe_fusion = BaselineSafeDepthFusion(
        decoder_layers,
        initial_query_weight=0.05,
        max_query_weight=0.5,
    )
    safe_output = safe_fusion(
        baseline_depth,
        raw_query_depth,
        layer_index=2,
    )
    assert torch.allclose(
        safe_output["query_weight"],
        torch.full_like(baseline_depth, 0.05),
        atol=1.0e-6,
    )
    assert torch.allclose(
        safe_output["depth"],
        torch.full_like(baseline_depth, 20.5),
        atol=1.0e-6,
    )
    assert (safe_output["query_weight"] <= 0.5).all()

    gate = UncertaintyGeometryGate(
        hidden_dim,
        hidden_gate_dim=16,
        initial_stereo_weight=0.9,
        max_correction=10.0,
    )
    gate_output = gate(
        features[-1],
        final_refined["expected_depth"],
        final_refined["expected_depth"].detach() + 2.0,
        final_refined["entropy"],
        final_sampled["background_probability"],
        torch.zeros(batch_size, num_queries, 1),
        final_sampled["point_depth_variance"],
    )
    assert gate_output["depth"].shape == (
        batch_size,
        num_queries,
        1,
    )
    assert ((gate_output["stereo_weight"] > 0) & (
        gate_output["stereo_weight"] < 1
    )).all()

    objective = (
        gate_output["depth"].mean()
        + final_refined["entropy"].mean()
        + final_sampled["point_depth_variance"].mean()
        + safe_output["depth"].mean()
    )
    objective.backward()
    assert depth_logits.grad is not None
    assert torch.isfinite(depth_logits.grad).all()
    for feature in features:
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
    assert baseline_depth.grad is not None
    assert raw_query_depth.grad is not None
    assert safe_fusion.query_weight_logits.grad is not None
    assert torch.isfinite(safe_fusion.query_weight_logits.grad).all()
    print("查询深度、基线安全融合、概率归一化与反向传播测试通过")


if __name__ == "__main__":
    main()
