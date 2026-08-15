"""CPU tests for sparse epipolar refinement without CUDA extensions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "lib" / "models" / "monodetr" / "query_epipolar.py"
)
SPEC = importlib.util.spec_from_file_location(
    "qecr_query_epipolar_standalone",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
QueryEpipolarConsistencyRefiner = (
    MODULE.QueryEpipolarConsistencyRefiner
)


def make_inputs(batch_size=2, queries=6, channels=8, height=6, width=12):
    torch.manual_seed(7)
    left = torch.randn(
        batch_size,
        channels,
        height,
        width,
        requires_grad=True,
    )
    right = torch.randn(
        batch_size,
        channels,
        height,
        width,
        requires_grad=True,
    )
    points = torch.rand(batch_size, queries, 2) * 0.4 + 0.3
    bins = torch.tensor(
        [8.0, 10.0, 12.0, 14.0],
    ).view(1, 1, -1).expand(batch_size, queries, -1)
    logits = torch.tensor(
        [-2.0, 1.5, 0.5, -2.0],
    ).view(1, 1, -1).expand(batch_size, queries, -1).clone()
    probabilities = logits.softmax(dim=-1)
    indices = torch.arange(4).view(1, 1, -1).expand_as(probabilities)
    stereo_fb = torch.tensor([42.0, 42.0])
    image_width = torch.tensor([48.0, 48.0])
    direction = torch.tensor([-1.0, 1.0])
    return {
        "left_features": left,
        "right_features": right,
        "query_points": points,
        "local_logits": logits,
        "local_bins": bins,
        "local_indices": indices,
        "full_probabilities": probabilities,
        "stereo_fb": stereo_fb,
        "image_width": image_width,
        "stereo_direction": direction,
    }


def test_shapes_gradients_and_probability_mass():
    module = QueryEpipolarConsistencyRefiner(
        num_samples=5,
        search_radius=2.0,
        bidirectional=True,
    )
    inputs = make_inputs()
    output = module(**inputs)
    expected_shapes = {
        "probabilities": (2, 6, 4),
        "expected_depth": (2, 6, 1),
        "epipolar_depth": (2, 6, 1),
        "confidence": (2, 6, 1),
        "cycle_error": (2, 6, 1),
        "valid": (2, 6, 1),
    }
    for key, shape in expected_shapes.items():
        assert output[key].shape == shape, (key, output[key].shape, shape)
        assert torch.isfinite(output[key]).all(), key
    assert torch.allclose(
        output["probabilities"].sum(dim=-1),
        torch.ones(2, 6),
        atol=1.0e-6,
    )
    assert ((output["confidence"] >= 0.0) & (
        output["confidence"] <= 1.0
    )).all()

    loss = output["expected_depth"].mean() + output["confidence"].mean()
    loss.backward()
    assert inputs["left_features"].grad is not None
    assert inputs["right_features"].grad is not None
    assert torch.isfinite(inputs["left_features"].grad).all()
    assert torch.isfinite(inputs["right_features"].grad).all()


def test_out_of_view_queries_fall_back_to_original_distribution():
    module = QueryEpipolarConsistencyRefiner(
        num_samples=5,
        search_radius=1.0,
        bidirectional=True,
    )
    inputs = make_inputs(batch_size=1, queries=2)
    inputs["query_points"] = torch.tensor(
        [[[0.01, 0.5], [0.02, 0.6]]]
    )
    inputs["stereo_fb"] = torch.tensor([5000.0])
    inputs["image_width"] = torch.tensor([48.0])
    inputs["stereo_direction"] = torch.tensor([-1.0])
    original = inputs["local_logits"].softmax(dim=-1)
    output = module(**inputs)
    assert torch.equal(output["valid"], torch.zeros_like(output["valid"]))
    assert torch.equal(
        output["confidence"],
        torch.zeros_like(output["confidence"]),
    )
    assert torch.allclose(
        output["probabilities"],
        original,
        atol=1.0e-6,
    )


def main():
    test_shapes_gradients_and_probability_mass()
    test_out_of_view_queries_fall_back_to_original_distribution()
    print("QECR 稀疏极线模块 CPU 测试通过")


if __name__ == "__main__":
    main()

