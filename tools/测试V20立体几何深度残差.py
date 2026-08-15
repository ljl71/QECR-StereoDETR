"""Unit tests for the V20 baseline-preserving metric-depth residual head."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.models.monodetr.geometry_depth_residual import (  # noqa: E402
    QueryGeometryDepthResidual,
)
from lib.models.monodetr.stereodetr import (  # noqa: E402
    freeze_except_parameter_prefixes,
)


def build(use_geometry: bool) -> QueryGeometryDepthResidual:
    return QueryGeometryDepthResidual(
        query_dim=8,
        hidden_dim=4,
        max_correction=3.0,
        use_geometry_prior=use_geometry,
        detach_inputs=True,
    )


def test_zero_initialization_and_equal_capacity() -> None:
    torch.manual_seed(7)
    query = torch.randn(2, 5, 8)
    stereo = torch.rand(2, 5, 1) * 40.0 + 2.0
    geometry = stereo + torch.randn_like(stereo)
    query_only = build(False)
    geometry_aware = build(True)
    count_a = sum(parameter.numel() for parameter in query_only.parameters())
    count_b = sum(parameter.numel() for parameter in geometry_aware.parameters())
    assert count_a == count_b
    for module in (query_only, geometry_aware):
        output = module(query, stereo, geometry)
        assert torch.equal(output["depth"], stereo)
        assert torch.count_nonzero(output["residual"]) == 0
    assert torch.count_nonzero(
        query_only(query, stereo, geometry)["geometry_context"]
    ) == 0
    expected = ((geometry - stereo) / stereo.abs().clamp_min(1.0)).clamp(
        -1.0, 1.0
    )
    actual = geometry_aware(query, stereo, geometry)["geometry_context"]
    assert torch.allclose(actual, expected)


def test_bound_and_backward() -> None:
    torch.manual_seed(11)
    module = build(True)
    nn.init.normal_(module.layers[-1].weight, mean=0.0, std=0.2)
    nn.init.constant_(module.layers[-1].bias, 0.1)
    query = torch.randn(2, 5, 8)
    stereo = torch.rand(2, 5, 1) * 20.0 + 1.0
    geometry = stereo + torch.randn_like(stereo) * 2.0
    output = module(query, stereo, geometry)
    assert float(output["residual"].abs().max()) <= 3.0 + 1.0e-6
    loss = output["depth"].square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in module.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_freeze_scope() -> None:
    class Dummy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(3, 3)
            self.quality_head = nn.Linear(3, 1)
            self.depth_residual_head = build(True)

    model = Dummy()
    trainable = freeze_except_parameter_prefixes(
        model, ["depth_residual_head."]
    )
    assert trainable
    assert all(name.startswith("depth_residual_head.") for name in trainable)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.startswith("depth_residual_head.")


def main() -> None:
    test_zero_initialization_and_equal_capacity()
    test_bound_and_backward()
    test_freeze_scope()
    print(
        "V20深度残差测试通过：零初始化逐位恢复V09、A/B参数量一致、"
        "几何差值正确、有界残差与反向传播正常、冻结范围未越界"
    )


if __name__ == "__main__":
    main()

