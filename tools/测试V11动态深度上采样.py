"""Unit tests for V11 baseline-preserving dynamic depth upsampling."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.optimizer_helper import build_optimizer  # noqa: E402
from lib.models.monodetr.depth_predictor.depth_predictor_lightstereo import (  # noqa: E402
    BaselinePreservingDynamicUpsample,
)
from lib.models.monodetr.stereodetr import (  # noqa: E402
    freeze_except_parameter_prefixes,
)


def test_zero_offset_matches_bilinear() -> None:
    torch.manual_seed(444)
    for shape in ((2, 8, 5, 7), (1, 8, 3, 4), (1, 8, 2, 5)):
        feature = torch.randn(*shape)
        module = BaselinePreservingDynamicUpsample(
            channels=8, scale_factor=2, max_offset=0.5
        )
        actual = module(feature)
        expected = F.interpolate(
            feature,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        maximum_error = float((actual - expected).abs().max())
        assert maximum_error <= 1.0e-5, (shape, maximum_error)


def test_offset_bounds_and_backward() -> None:
    torch.manual_seed(445)
    feature = torch.randn(2, 8, 5, 7, requires_grad=True)
    module = BaselinePreservingDynamicUpsample(
        channels=8, scale_factor=2, max_offset=0.5
    )
    nn.init.normal_(module.offset_predictor.weight, mean=0.0, std=0.01)
    offsets = module.predict_offsets(feature)
    assert offsets.shape == (2, 2, 10, 14)
    assert float(offsets.abs().max()) <= 0.5 + 1.0e-6
    output = module(feature)
    loss = output.square().mean()
    loss.backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    gradient = module.offset_predictor.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


class _DummyDepthPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.depth_classifier = nn.Sequential(
            BaselinePreservingDynamicUpsample(8),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.BatchNorm2d(4),
        )
        self.other = nn.Conv2d(8, 8, 1)


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.depth_predictor = _DummyDepthPredictor()
        self.quality_head = nn.Linear(8, 1)


def test_training_scope_and_learning_rate_multiplier() -> None:
    model = _DummyModel()
    trainable = freeze_except_parameter_prefixes(
        model, ["depth_predictor.depth_classifier."]
    )
    assert trainable
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.startswith(
            "depth_predictor.depth_classifier."
        )

    optimizer = build_optimizer(
        {
            "type": "adamw",
            "lr": 2.0e-5,
            "weight_decay": 1.0e-4,
            "parameter_lr_multipliers": {"offset_predictor": 10.0},
        },
        model,
    )
    learning_rates = {float(group["lr"]) for group in optimizer.param_groups}
    assert learning_rates == {2.0e-5, 2.0e-4}, learning_rates


def test_invalid_settings() -> None:
    try:
        BaselinePreservingDynamicUpsample(8, max_offset=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero max_offset must be rejected")

    try:
        freeze_except_parameter_prefixes(_DummyModel(), ["missing."])
    except ValueError:
        pass
    else:
        raise AssertionError("unmatched training scope must be rejected")


def main() -> None:
    test_zero_offset_matches_bilinear()
    test_offset_bounds_and_backward()
    test_training_scope_and_learning_rate_multiplier()
    test_invalid_settings()
    print(
        "V11动态深度上采样测试通过：零偏移恢复双线性、偏移有界、"
        "反向传播与分组学习率正常"
    )


if __name__ == "__main__":
    main()
