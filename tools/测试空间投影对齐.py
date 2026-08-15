"""CPU/CUDA-independent checks for V08 training-only geometry alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.models.monodetr.geometry_alignment import (  # noqa: E402
    corner_alignment_loss,
    cuboid_corners,
    decode_camera_geometry,
    decode_predicted_alpha,
    decode_target_alpha,
    encoded_boxes_to_original_xyxy,
    progressive_weight,
    project_corners,
    projected_enclosing_boxes,
    projection_alignment_loss,
)
from lib.models.monodetr.stereodetr import SetCriterion  # noqa: E402


class DummyMatcher:
    def __call__(self, outputs, targets, group_num=1):
        indices = []
        for target in targets:
            count = len(target["labels"])
            index = torch.arange(count, dtype=torch.long)
            indices.append((index, index))
        return indices, indices


def make_inputs(perturb=False):
    encoded = torch.tensor(
        [[0.52, 0.55, 0.08, 0.09, 0.18, 0.20]],
        dtype=torch.float32,
        requires_grad=perturb,
    )
    depth = torch.tensor([22.0], dtype=torch.float32, requires_grad=perturb)
    dimensions = torch.tensor(
        [[1.55, 1.65, 3.90]], dtype=torch.float32, requires_grad=perturb
    )
    heading = torch.zeros((1, 24), dtype=torch.float32)
    heading[0, 2] = 4.0
    heading[0, 14] = 0.07
    heading.requires_grad_(perturb)
    projection = torch.tensor(
        [[[700.0, 0.0, 640.0, 0.0],
          [0.0, 700.0, 194.0, 0.0],
          [0.0, 0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    cropped = torch.tensor([[1280.0, 288.0]], dtype=torch.float32)
    original = torch.tensor([[1280.0, 388.0]], dtype=torch.float32)
    upper = torch.tensor([100.0], dtype=torch.float32)
    return encoded, depth, dimensions, heading, projection, cropped, original, upper


def test_exact_and_perturbed_geometry():
    encoded, depth, dimensions, heading, projection, cropped, original, upper = make_inputs()
    alpha = decode_predicted_alpha(heading)
    geometry = decode_camera_geometry(
        encoded, depth, dimensions, alpha, projection, cropped, original, upper
    )
    corners = cuboid_corners(
        geometry["center"], geometry["dimensions"], geometry["rotation_y"]
    )
    exact = corner_alignment_loss(corners, corners)
    assert torch.allclose(exact, torch.zeros_like(exact), atol=1.0e-7)
    shifted = corners.clone()
    shifted[..., 2] += 2.0
    assert corner_alignment_loss(shifted, corners).item() > exact.item()

    points, visible = project_corners(corners, projection)
    assert visible.all()
    boxes = projected_enclosing_boxes(points, original)
    assert projection_alignment_loss(boxes, boxes).abs().max().item() < 1.0e-6
    expanded = boxes + torch.tensor([[-0.02, -0.01, 0.02, 0.01]])
    assert projection_alignment_loss(expanded, boxes).item() > 0.0


def test_schedule_and_target_heading():
    assert progressive_weight(0, 1, 4) == 0.0
    assert progressive_weight(1, 1, 4) == 0.25
    assert progressive_weight(4, 1, 4) == 1.0
    target_alpha = decode_target_alpha(torch.tensor([[2]]), torch.tensor([[0.07]]))
    prediction = torch.zeros((1, 24))
    prediction[0, 2] = 4.0
    prediction[0, 14] = 0.07
    assert torch.allclose(decode_predicted_alpha(prediction), target_alpha)


def test_criterion_backward():
    encoded, depth, dimensions, heading, projection, cropped, original, upper = make_inputs(perturb=True)
    target_boxes = encoded.detach().clone()
    target_boxes[:, 0] -= 0.015
    target_depth = depth.detach().clone() - 1.0
    target_dimensions = dimensions.detach().clone() * torch.tensor([[1.0, 0.95, 1.05]])
    target_bin = torch.tensor([[2]], dtype=torch.long)
    target_res = torch.tensor([[0.02]], dtype=torch.float32)
    criterion = SetCriterion(
        num_classes=4,
        matcher=DummyMatcher(),
        weight_dict={},
        focal_alpha=0.25,
        losses=["geometry_alignment"],
        geometry_alignment_cfg={
            "enabled": True,
            "corner_enabled": True,
            "projection_enabled": True,
            "start_epoch": 1,
            "ramp_epochs": 4,
            "projection_boundary_weight": 0.25,
        },
    )
    outputs = {
        "pred_boxes": encoded.unsqueeze(0),
        "pred_depth": torch.stack((depth, torch.zeros_like(depth)), dim=-1).unsqueeze(0),
        "pred_3d_dim": dimensions.unsqueeze(0),
        "pred_angle": heading.unsqueeze(0),
        "geometry_calibs": projection,
        "geometry_img_sizes": cropped,
        "geometry_img_sizes_ori": original,
        "geometry_img_sizes_upper": upper,
    }
    targets = [{
        "labels": torch.tensor([[1]], dtype=torch.long),
        "boxes_3d": target_boxes,
        "depth": target_depth.unsqueeze(-1),
        "size_3d": target_dimensions,
        "heading_bin": target_bin,
        "heading_res": target_res,
    }]
    index = [(torch.tensor([0]), torch.tensor([0]))]
    criterion.set_training_epoch(0)
    cold = criterion.loss_geometry_alignment(outputs, targets, index, index, 1.0)
    assert cold["loss_geometry_corner"].item() == 0.0
    assert cold["loss_geometry_projection"].item() == 0.0
    criterion.set_training_epoch(1)
    losses = criterion.loss_geometry_alignment(outputs, targets, index, index, 1.0)
    total = losses["loss_geometry_corner"] + losses["loss_geometry_projection"]
    assert torch.isfinite(total) and total.item() > 0.0
    total.backward()
    for name, tensor in {
        "boxes": encoded,
        "depth": depth,
        "dimensions": dimensions,
        "heading": heading,
    }.items():
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum().item() > 0.0, name

    xyxy = encoded_boxes_to_original_xyxy(target_boxes, cropped, original, upper)
    assert (xyxy[:, 2:] > xyxy[:, :2]).all()


def main():
    test_exact_and_perturbed_geometry()
    test_schedule_and_target_heading()
    test_criterion_backward()
    print("V08空间角点、左目投影、分阶段权重与反向传播测试通过")


if __name__ == "__main__":
    main()
