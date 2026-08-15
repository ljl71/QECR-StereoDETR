"""Unit tests for the training-only V22 axial depth objective."""

from __future__ import annotations

import math
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "lib" / "models" / "monodetr" / "axial_depth_iou.py"
SPEC = importlib.util.spec_from_file_location("v22_axial_depth_iou", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load {}".format(MODULE_PATH))
AXIAL_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AXIAL_MODULE)
axial_half_extent_hwl = AXIAL_MODULE.axial_half_extent_hwl
depth_only_axial_giou_loss = AXIAL_MODULE.depth_only_axial_giou_loss
generalized_interval_iou_loss = AXIAL_MODULE.generalized_interval_iou_loss


def test_half_extent_geometry() -> None:
    dimensions = torch.tensor([[1.5, 2.0, 4.0]])
    yaw = torch.tensor([0.0, math.pi / 2.0])
    repeated = dimensions.expand(2, -1)
    half_extent = axial_half_extent_hwl(repeated, yaw)
    assert torch.allclose(half_extent, torch.tensor([1.0, 2.0]), atol=1.0e-6)


def test_interval_giou_order_and_disjoint_gradient() -> None:
    half = torch.tensor([1.0])
    identical = generalized_interval_iou_loss(
        torch.tensor([10.0]), half, torch.tensor([10.0]), half
    )
    near = generalized_interval_iou_loss(
        torch.tensor([10.5]), half, torch.tensor([10.0]), half
    )
    far_depth = torch.tensor([14.0], requires_grad=True)
    far = generalized_interval_iou_loss(
        far_depth, half, torch.tensor([10.0]), half
    )
    assert torch.allclose(identical, torch.zeros_like(identical), atol=1.0e-6)
    assert float(identical) < float(near) < float(far)
    far.sum().backward()
    assert far_depth.grad is not None
    assert torch.isfinite(far_depth.grad).all()
    assert float(far_depth.grad.abs().max()) > 0.0


def test_depth_only_gradient_boundary() -> None:
    predicted_depth = torch.tensor([11.0], requires_grad=True)
    predicted_dimensions = torch.tensor(
        [[1.5, 1.7, 3.9]], requires_grad=True
    )
    predicted_yaw = torch.tensor([0.2], requires_grad=True)
    target_dimensions = torch.tensor(
        [[1.6, 1.8, 4.0]], requires_grad=True
    )
    target_yaw = torch.tensor([0.1], requires_grad=True)
    loss = depth_only_axial_giou_loss(
        predicted_depth=predicted_depth,
        target_depth=torch.tensor([10.0]),
        predicted_dimensions_hwl=predicted_dimensions,
        target_dimensions_hwl=target_dimensions,
        predicted_rotation_y=predicted_yaw,
        target_rotation_y=target_yaw,
    ).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert predicted_depth.grad is not None
    assert float(predicted_depth.grad.abs().max()) > 0.0
    assert predicted_dimensions.grad is None
    assert predicted_yaw.grad is None
    assert target_dimensions.grad is None
    assert target_yaw.grad is None


def main() -> None:
    test_half_extent_geometry()
    test_interval_giou_order_and_disjoint_gradient()
    test_depth_only_gradient_boundary()
    print(
        "V22轴向半长、重叠/不相交GIoU梯度及深度独占反向传播测试通过"
    )


if __name__ == "__main__":
    main()
