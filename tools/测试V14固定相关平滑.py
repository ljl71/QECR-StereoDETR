#!/usr/bin/env python3
"""Test the parameter-free V14 correlation regularizer and its configs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402
from lib.models.monodetr.depth_predictor.depth_predictor_lightstereo import (  # noqa: E402
    spatially_regularize_correlation,
)


def main() -> None:
    torch.manual_seed(14)
    volume = torch.randn(2, 24, 7, 9, requires_grad=True)

    disabled = spatially_regularize_correlation(volume, enabled=False)
    assert disabled is volume
    zero_blend = spatially_regularize_correlation(
        volume, enabled=True, kernel_size=3, blend=0.0
    )
    assert zero_blend is volume

    constant = torch.ones(1, 24, 5, 6)
    constant_smoothed = spatially_regularize_correlation(
        constant, enabled=True, kernel_size=3, blend=1.0
    )
    assert torch.equal(constant, constant_smoothed)

    impulse = torch.zeros(1, 1, 5, 5)
    impulse[0, 0, 2, 2] = 9.0
    impulse_smoothed = spatially_regularize_correlation(
        impulse, enabled=True, kernel_size=3, blend=1.0
    )
    expected = torch.zeros_like(impulse)
    expected[0, 0, 1:4, 1:4] = 1.0
    assert torch.allclose(impulse_smoothed, expected)

    output = spatially_regularize_correlation(
        volume, enabled=True, kernel_size=3, blend=1.0
    )
    assert output.shape == volume.shape
    output.square().mean().backward()
    assert volume.grad is not None
    assert torch.isfinite(volume.grad).all()

    control = load_config(
        ROOT / "versions" / "V14O_零训练原始相关复评" / "config.yaml"
    )
    candidate = load_config(
        ROOT / "versions" / "V14S_零训练s4固定平滑" / "config.yaml"
    )
    for config, enabled in ((control, False), (candidate, True)):
        smoothing = config["model"]["correlation_smoothing"]
        assert bool(smoothing["enabled"]) is enabled
        assert int(smoothing["scale"]) == 4
        assert int(smoothing["kernel_size"]) == 3
        assert float(smoothing["blend"]) == 1.0
        assert int(smoothing["passes"]) == 1
        assert bool(config["model"]["quality_ranking"]["enabled"])
        assert float(config["model"]["quality_ranking"]["score_power"]) == 1.5

    print(
        "V14固定相关平滑测试通过：关闭逐位返回、常量/脉冲行为、"
        "反向传播与零训练配置均正常"
    )


if __name__ == "__main__":
    main()
