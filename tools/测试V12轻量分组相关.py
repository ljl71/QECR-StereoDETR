"""Unit tests for the V12 baseline-preserving group-wise correlation gate."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.optimizer_helper import build_optimizer  # noqa: E402
from lib.models.monodetr.depth_predictor.depth_predictor_lightstereo import (  # noqa: E402
    BaselinePreservingGroupwiseCorrelation,
    correlation_volume,
    correlation_volume_flip,
)
from lib.models.monodetr.stereodetr import (  # noqa: E402
    freeze_except_parameter_prefixes,
)


def test_zero_gate_matches_baseline() -> None:
    torch.manual_seed(1201)
    left = torch.randn(2, 32, 7, 13)
    right = torch.randn(2, 32, 7, 13)
    module = BaselinePreservingGroupwiseCorrelation(
        channels=32, num_groups=8, gate_kernel_size=3
    )
    weights = module.gate_weights(left)
    expected_weights = torch.full_like(weights, 1.0 / 8.0)
    assert torch.equal(weights, expected_weights)

    for flip in (False, True):
        actual = module(left, right, max_disp=6, flip=flip)
        expected = (
            correlation_volume_flip(left, right, 6)
            if flip
            else correlation_volume(left, right, 6)
        )
        maximum_error = float((actual - expected).abs().max())
        assert maximum_error <= 2.0e-6, (flip, maximum_error)


def test_gate_backward_and_nonuniformity() -> None:
    torch.manual_seed(1202)
    left = torch.randn(2, 32, 5, 9, requires_grad=True)
    right = torch.randn(2, 32, 5, 9, requires_grad=True)
    module = BaselinePreservingGroupwiseCorrelation(32, 8, 3)
    nn.init.normal_(module.group_logits.weight, mean=0.0, std=0.02)
    output = module(left, right, max_disp=5)
    loss = output.square().mean()
    loss.backward()
    assert output.shape == (2, 5, 5, 9)
    assert left.grad is not None and torch.isfinite(left.grad).all()
    assert right.grad is not None and torch.isfinite(right.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert float(module.group_logits.weight.grad.abs().sum()) > 0.0
    assert float(module.context.weight.grad.abs().sum()) > 0.0


def _reference_groupwise(module, left, right, max_disp, flip):
    """Slow pre-optimization formula used to prove algebraic equivalence."""
    weights = module.gate_weights(left)
    batch, _, height, width = left.shape
    output = left.new_zeros(batch, max_disp, height, width)
    for disparity in range(max_disp):
        if disparity == 0:
            output[:, disparity] = module._compress(
                left * right, weights
            )
        elif flip:
            product = (
                left[:, :, :, :-disparity]
                * right[:, :, :, disparity:]
            )
            output[:, disparity, :, disparity:] = module._compress(
                product, weights[:, :, :, :-disparity]
            )
        else:
            product = (
                left[:, :, :, disparity:]
                * right[:, :, :, :-disparity]
            )
            output[:, disparity, :, disparity:] = module._compress(
                product, weights[:, :, :, disparity:]
            )
    return output


def test_optimized_formula_matches_groupwise_reference() -> None:
    torch.manual_seed(1203)
    left = torch.randn(2, 32, 6, 11)
    right = torch.randn(2, 32, 6, 11)
    module = BaselinePreservingGroupwiseCorrelation(32, 8, 3)
    nn.init.normal_(module.group_logits.weight, mean=0.0, std=0.08)
    nn.init.normal_(module.group_logits.bias, mean=0.0, std=0.03)
    for flip in (False, True):
        actual = module(left, right, max_disp=6, flip=flip)
        expected = _reference_groupwise(
            module, left, right, max_disp=6, flip=flip
        )
        maximum_error = float((actual - expected).abs().max())
        assert maximum_error <= 2.0e-6, (flip, maximum_error)


class _DummyDepthPredictor(nn.Module):
    def __init__(self, with_gate: bool) -> None:
        super().__init__()
        self.cost_agg = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8)
        )
        if with_gate:
            self.groupwise_correlation_s4 = (
                BaselinePreservingGroupwiseCorrelation(8, 4, 3)
            )
        else:
            self.groupwise_correlation_s4 = None
        self.other = nn.Conv2d(8, 8, 1)


class _DummyModel(nn.Module):
    def __init__(self, with_gate: bool) -> None:
        super().__init__()
        self.depth_predictor = _DummyDepthPredictor(with_gate)
        self.quality_head = nn.Linear(8, 1)


def test_training_scope_and_learning_rate() -> None:
    control = _DummyModel(with_gate=False)
    candidate = _DummyModel(with_gate=True)
    control_names = freeze_except_parameter_prefixes(
        control, ["depth_predictor.cost_agg."]
    )
    candidate_names = freeze_except_parameter_prefixes(
        candidate,
        [
            "depth_predictor.cost_agg.",
            "depth_predictor.groupwise_correlation_s4.",
        ],
    )
    non_gate_candidate = [
        name for name in candidate_names
        if "groupwise_correlation_s4" not in name
    ]
    assert control_names == non_gate_candidate
    assert not any(name.startswith("quality_head.") for name in candidate_names)

    optimizer = build_optimizer(
        {
            "type": "adamw",
            "lr": 2.0e-5,
            "weight_decay": 1.0e-4,
            "parameter_lr_multipliers": {
                "groupwise_correlation_s4": 10.0
            },
        },
        candidate,
    )
    learning_rates = {float(group["lr"]) for group in optimizer.param_groups}
    assert learning_rates == {2.0e-5, 2.0e-4}, learning_rates


def test_invalid_settings() -> None:
    for arguments in (
        {"channels": 30, "num_groups": 8},
        {"channels": 32, "num_groups": 1},
        {"channels": 32, "num_groups": 8, "gate_kernel_size": 2},
    ):
        try:
            BaselinePreservingGroupwiseCorrelation(**arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid V12 settings must be rejected")


def main() -> None:
    test_zero_gate_matches_baseline()
    test_gate_backward_and_nonuniformity()
    test_optimized_formula_matches_groupwise_reference()
    test_training_scope_and_learning_rate()
    test_invalid_settings()
    print(
        "V12轻量分组相关测试通过：零初始化恢复原相关、翻转路径等价、"
        "逐视差压缩与反向传播正常、训练白名单和分组学习率正确"
    )


if __name__ == "__main__":
    main()
