"""Unit and configuration checks for the V10 depth readout diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402
from lib.models.monodetr.depth_predictor.depth_readout import (  # noqa: E402
    ForegroundTopKDepthReadout,
)


def test_disabled_is_bitwise_original() -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, 6, 3, 4)
    bins = torch.tensor([2.0, 4.0, 8.0, 16.0, 32.0, 60.0])
    probabilities = F.softmax(logits, dim=1)
    expected = (probabilities * bins.reshape(1, -1, 1, 1)).sum(dim=1)
    actual = ForegroundTopKDepthReadout(False, 2)(
        logits,
        bins,
        depth_probs=probabilities,
    )
    assert torch.equal(actual, expected)


def test_all_foreground_bins_recover_original() -> None:
    torch.manual_seed(11)
    logits = torch.randn(2, 6, 3, 4, dtype=torch.float64)
    bins = torch.tensor(
        [2.0, 4.0, 8.0, 16.0, 32.0, 60.0],
        dtype=torch.float64,
    )
    probabilities = F.softmax(logits, dim=1)
    expected = (probabilities * bins.reshape(1, -1, 1, 1)).sum(dim=1)
    actual = ForegroundTopKDepthReadout(True, 5)(
        logits,
        bins,
        depth_probs=probabilities,
    )
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_background_probability_is_preserved() -> None:
    logits = torch.tensor([[[[0.0]], [[-1.0]], [[-2.0]], [[20.0]]]])
    bins = torch.tensor([5.0, 15.0, 30.0, 60.0])
    depth = ForegroundTopKDepthReadout(True, 2)(logits, bins)
    assert abs(float(depth.item()) - 60.0) < 1.0e-6


def test_topk_concentrates_multimodal_foreground() -> None:
    logits = torch.tensor(
        [[[[5.0]], [[4.0]], [[1.0]], [[0.0]], [[-20.0]]]],
        requires_grad=True,
    )
    bins = torch.tensor([5.0, 10.0, 30.0, 50.0, 60.0])
    baseline = ForegroundTopKDepthReadout(False, 2)(logits, bins)
    concentrated = ForegroundTopKDepthReadout(True, 2)(logits, bins)
    assert 5.0 < float(concentrated.item()) < 10.0
    assert float(concentrated.item()) < float(baseline.item())
    concentrated.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_invalid_topk_fails_loudly() -> None:
    logits = torch.randn(1, 5, 1, 1)
    bins = torch.arange(5, dtype=torch.float32)
    try:
        ForegroundTopKDepthReadout(True, 5)(logits, bins)
    except ValueError as error:
        assert "exceeds" in str(error)
    else:
        raise AssertionError("top_k larger than foreground bins must fail")


def test_v10_configs() -> None:
    expected = {
        "V10O_全分布深度读出控制": (False, 2),
        "V10A_前景TopK2深度读出": (True, 2),
        "V10B_前景TopK4深度读出": (True, 4),
    }
    for version, values in expected.items():
        config = load_config(ROOT / "versions" / version / "config.yaml")
        readout = config["model"]["depth_readout"]
        assert (bool(readout["enabled"]), int(readout["top_k"])) == values
        assert bool(config["model"]["quality_ranking"]["enabled"])
        assert float(config["model"]["quality_ranking"]["score_power"]) == 1.5
        assert not bool(config["model"]["query_depth"]["enabled"])
        assert not bool(config["model"]["query_epipolar"]["enabled"])


def main() -> None:
    test_disabled_is_bitwise_original()
    test_all_foreground_bins_recover_original()
    test_background_probability_is_preserved()
    test_topk_concentrates_multimodal_foreground()
    test_invalid_topk_fails_loudly()
    test_v10_configs()
    module = ForegroundTopKDepthReadout(True, 2)
    assert not module.state_dict()
    print(
        "V10前景Top-K深度读出测试通过：关闭时逐位恢复原始期望、"
        "K覆盖全部前景时数学等价、背景质量保留且反向传播正常"
    )


if __name__ == "__main__":
    main()

