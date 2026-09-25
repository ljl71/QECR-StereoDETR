"""Unit tests for V23 RDSA/GPSD correlation pre-aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.models.monodetr.depth_predictor.depth_predictor_lightstereo import (  # noqa: E402
    GroupPreservingSpatialDisparityPreAggregation,
    ResidualDisparitySpaceMicroAggregation,
    correlation_volume,
    correlation_volume_flip,
    groupwise_correlation_volume,
)


def parameter_count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_groupwise_mean_equivalence() -> None:
    torch.manual_seed(23)
    left = torch.randn(2, 8, 5, 9)
    right = torch.randn_like(left)
    for flip, baseline_function in (
        (False, correlation_volume),
        (True, correlation_volume_flip),
    ):
        grouped = groupwise_correlation_volume(
            left, right, max_disp=5, num_groups=4, flip=flip
        )
        baseline = baseline_function(left, right, 5)
        assert grouped.shape == (2, 4, 5, 5, 9)
        assert torch.allclose(grouped.mean(dim=1), baseline, atol=1.0e-6)


def test_rdsa_identity_parameters_and_backward() -> None:
    torch.manual_seed(23)
    module = ResidualDisparitySpaceMicroAggregation(hidden_channels=4)
    assert parameter_count(module) == 666
    baseline = torch.randn(2, 6, 5, 7, requires_grad=True)
    output = module(baseline)
    assert torch.equal(output, baseline)
    output.square().mean().backward()
    final_gamma = module.aggregate[-1].weight
    assert final_gamma.grad is not None
    assert torch.isfinite(final_gamma.grad).all()
    assert float(final_gamma.grad.abs().max()) > 0.0

    with torch.no_grad():
        final_gamma.fill_(0.1)
    changed = module(baseline.detach())
    assert not torch.equal(changed, baseline.detach())


def test_gpsd_identity_parameters_and_backward() -> None:
    torch.manual_seed(23)
    module = GroupPreservingSpatialDisparityPreAggregation(
        num_groups=4,
        num_disparities=6,
        expansion_ratio=2,
    )
    # For D=6: in=24, hidden=48. Conv/BN parameter count is deterministic.
    assert parameter_count(module) == 2076
    baseline = torch.randn(2, 6, 5, 7, requires_grad=True)
    groups = torch.randn(2, 4, 6, 5, 7, requires_grad=True)
    output = module(baseline, groups)
    assert torch.equal(output, baseline)
    output.square().mean().backward()
    final_gamma = module.pre_aggregate[-1].weight
    assert final_gamma.grad is not None
    assert torch.isfinite(final_gamma.grad).all()
    assert float(final_gamma.grad.abs().max()) > 0.0

    full_size = GroupPreservingSpatialDisparityPreAggregation(
        num_groups=4,
        num_disparities=24,
        expansion_ratio=2,
    )
    assert parameter_count(full_size) == 25584
    assert parameter_count(full_size) / 18459785.0 < 0.0015


def test_invalid_shapes_are_rejected() -> None:
    module = GroupPreservingSpatialDisparityPreAggregation(
        num_groups=4, num_disparities=6
    )
    baseline = torch.randn(1, 6, 4, 4)
    wrong = torch.randn(1, 2, 6, 4, 4)
    try:
        module(baseline, wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("GPSD accepted a wrong group count")


def main() -> None:
    test_groupwise_mean_equivalence()
    test_rdsa_identity_parameters_and_backward()
    test_gpsd_identity_parameters_and_backward()
    test_invalid_shapes_are_rejected()
    print(
        "V23测试通过：分组均值恢复原相关，RDSA/GPSD零初始化恒等，"
        "参数量、反向传播与形状保护正常"
    )


if __name__ == "__main__":
    main()
