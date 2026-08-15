"""Run the QECR tensor tests against the runtime module."""

from QECR运行时引导 import install_runtime_module


MODULE = install_runtime_module()
QueryEpipolarConsistencyRefiner = (
    MODULE.QueryEpipolarConsistencyRefiner
)

import torch  # noqa: E402


def make_inputs(batch_size=2, queries=6, channels=8, height=6, width=12):
    torch.manual_seed(7)
    left = torch.randn(
        batch_size, channels, height, width, requires_grad=True
    )
    right = torch.randn(
        batch_size, channels, height, width, requires_grad=True
    )
    points = torch.rand(batch_size, queries, 2) * 0.4 + 0.3
    bins = torch.tensor(
        [8.0, 10.0, 12.0, 14.0]
    ).view(1, 1, -1).expand(batch_size, queries, -1)
    logits = torch.tensor(
        [-2.0, 1.5, 0.5, -2.0]
    ).view(1, 1, -1).expand(batch_size, queries, -1).clone()
    probabilities = logits.softmax(dim=-1)
    return {
        "left_features": left,
        "right_features": right,
        "query_points": points,
        "local_logits": logits,
        "local_bins": bins,
        "local_indices": torch.arange(4).view(
            1, 1, -1
        ).expand_as(probabilities),
        "full_probabilities": probabilities,
        "stereo_fb": torch.full((batch_size,), 42.0),
        "image_width": torch.full((batch_size,), 48.0),
        "stereo_direction": torch.tensor(
            [-1.0, 1.0][:batch_size]
        ),
    }


def test_shapes_gradients_and_mass():
    module = QueryEpipolarConsistencyRefiner(
        num_samples=5,
        search_radius=2.0,
        bidirectional=True,
    )
    inputs = make_inputs()
    output = module(**inputs)
    for key, shape in {
        "probabilities": (2, 6, 4),
        "expected_depth": (2, 6, 1),
        "epipolar_depth": (2, 6, 1),
        "confidence": (2, 6, 1),
        "cycle_error": (2, 6, 1),
        "valid": (2, 6, 1),
    }.items():
        assert output[key].shape == shape
        assert torch.isfinite(output[key]).all()
    assert torch.allclose(
        output["probabilities"].sum(dim=-1),
        torch.ones(2, 6),
        atol=1.0e-6,
    )
    assert ((output["confidence"] >= 0.0) & (
        output["confidence"] <= 1.0
    )).all()
    (
        output["expected_depth"].mean()
        + output["confidence"].mean()
    ).backward()
    assert inputs["left_features"].grad is not None
    assert inputs["right_features"].grad is not None
    assert torch.isfinite(inputs["left_features"].grad).all()
    assert torch.isfinite(inputs["right_features"].grad).all()


def test_invalid_fallback():
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


if __name__ == "__main__":
    test_shapes_gradients_and_mass()
    test_invalid_fallback()
    print("QECR 运行时模块前向、反向与越界回退测试通过")

