"""Validate SGBM candidate packing, soft LRC and probability-mass loss."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset  # noqa: E402

# Load the loss file directly so the local CPU test does not require the
# compiled MultiScaleDeformableAttention extension.  AutoDL's full import test
# still verifies the normal package chain separately.
LOSS_PATH = (
    ROOT
    / "lib"
    / "models"
    / "monodetr"
    / "depth_predictor"
    / "disparityloss.py"
)
LOSS_SPEC = importlib.util.spec_from_file_location(
    "qecr_disparityloss_test_module", LOSS_PATH
)
LOSS_MODULE = importlib.util.module_from_spec(LOSS_SPEC)
LOSS_SPEC.loader.exec_module(LOSS_MODULE)
DisparityLoss = LOSS_MODULE.DisparityLoss
MultiCandidateDisparityLoss = LOSS_MODULE.MultiCandidateDisparityLoss


def _dataset_stub(matcher_mode="SGBM", lrc=True):
    dataset = object.__new__(KITTI_Dataset)
    dataset.matcher_mode = matcher_mode
    dataset.use_multi_candidate_disparity = True
    dataset.disparity_lrc_soft_weight = bool(lrc)
    dataset.disparity_lrc_temperature = 1.0
    dataset.disparity_lrc_confidence_floor = 0.25
    dataset.disparity_cell_size = 4
    return dataset


def test_candidate_packing():
    fixed = np.arange(8 * 12, dtype=np.uint16).reshape(8, 12)
    confidence = np.ones_like(fixed, dtype=np.float32)
    scalar, candidates, weights = KITTI_Dataset._pack_disparity_cells(
        fixed, confidence, 4
    )
    assert scalar.shape == (2, 3)
    assert candidates.shape == (16, 2, 3)
    assert weights.shape == candidates.shape
    expected = fixed.reshape(2, 4, 3, 4).max(axis=(1, 3))
    assert np.array_equal(scalar, expected)
    assert np.array_equal(scalar, candidates.max(axis=0))


def test_actual_matchers_and_lrc():
    rng = np.random.default_rng(20260808)
    left = rng.integers(0, 256, size=(64, 320, 3), dtype=np.uint8)
    right = np.roll(left, -6, axis=1)
    left_image = Image.fromarray(left, mode="RGB")
    right_image = Image.fromarray(right, mode="RGB")

    outputs = {}
    for matcher_mode in ("StereoBM", "SGBM"):
        dataset = _dataset_stub(matcher_mode, lrc=True)
        scalar, candidates, weights = dataset.get_disparty_P2_onfly(
            left_image, right_image, False, False
        )
        assert scalar.shape == (16, 80)
        assert candidates.shape == (16, 16, 80)
        assert weights.shape == candidates.shape
        assert np.array_equal(scalar, candidates.max(axis=0))
        assert np.isfinite(weights).all()
        assert float(weights.min()) >= 0.0
        assert float(weights.max()) <= 1.0
        outputs[matcher_mode] = scalar
    assert not np.array_equal(outputs["StereoBM"], outputs["SGBM"])

    # Flipped augmentation must keep shapes and positive disparity magnitudes.
    dataset = _dataset_stub("SGBM", lrc=True)
    scalar, candidates, weights = dataset.get_disparty_P2_onfly(
        left_image.transpose(Image.FLIP_LEFT_RIGHT),
        right_image.transpose(Image.FLIP_LEFT_RIGHT),
        True,
        False,
    )
    assert scalar.shape == (16, 80)
    assert np.all(candidates >= 0)
    assert np.isfinite(weights).all()


def test_probability_mass_loss():
    torch.manual_seed(9)
    logits = torch.randn(2, 96, 8, 12, requires_grad=True)
    candidates = torch.rand(2, 16, 8, 12) * 80.0 + 1.0
    weights = torch.rand_like(candidates) * 0.75 + 0.25
    candidates[:, :, 0, 0] = 0.0
    weights[:, :, 0, 0] = 0.0

    criterion = MultiCandidateDisparityLoss(
        maxdisp=96,
        variance=0.5,
        smoothing_radius=4,
        use_soft_confidence=True,
    )
    assert sum(parameter.numel() for parameter in criterion.parameters()) == 0
    target, valid, confidence = criterion.build_target_probability(
        candidates, weights, logits.dtype
    )
    probability_sum = target.sum(dim=1, keepdim=True)
    assert torch.allclose(
        probability_sum[valid],
        torch.ones_like(probability_sum[valid]),
        atol=1.0e-5,
    )
    assert not bool(valid[:, :, 0, 0].any())
    assert torch.isfinite(confidence).all()

    permutation = torch.randperm(candidates.shape[1])
    permuted_target, _, _ = criterion.build_target_probability(
        candidates[:, permutation], weights[:, permutation], logits.dtype
    )
    assert torch.allclose(target, permuted_target, atol=1.0e-6)

    loss = criterion(logits, candidates, weights)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0

    scalar_loss = DisparityLoss(maxdisp=96)
    scalar_label = torch.full((1, 2, 3), 12.5)
    scalar_logits = torch.randn(1, 96, 2, 3, requires_grad=True)
    baseline_value = scalar_loss(scalar_logits, scalar_label)
    assert torch.isfinite(baseline_value)
    baseline_value.backward()
    assert torch.isfinite(scalar_logits.grad).all()


def main():
    test_candidate_packing()
    test_actual_matchers_and_lrc()
    test_probability_mass_loss()
    print(
        "多候选视差监督测试通过：SGBM生效、16候选不丢失、"
        "LRC为软权重、概率质量归一化且反向传播正常"
    )


if __name__ == "__main__":
    main()
