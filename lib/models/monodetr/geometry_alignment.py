"""Training-only spatial and projection alignment for StereoDETR.

The module decodes matched query attributes into camera-frame cuboids. It has
no parameters and is never used by the inference decoder.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def decode_predicted_alpha(heading: torch.Tensor) -> torch.Tensor:
    """Decode the hard heading bin with a differentiable residual."""
    if heading.shape[-1] != 24:
        raise ValueError("heading must contain 12 bin logits and 12 residuals")
    bin_index = heading[..., :12].argmax(dim=-1, keepdim=True)
    residual = heading[..., 12:24].gather(-1, bin_index).squeeze(-1)
    center = bin_index.squeeze(-1).to(heading.dtype) * (2.0 * math.pi / 12.0)
    return wrap_angle(center + residual)


def decode_target_alpha(heading_bin: torch.Tensor, heading_res: torch.Tensor) -> torch.Tensor:
    center = heading_bin.reshape(-1).to(heading_res.dtype) * (2.0 * math.pi / 12.0)
    return wrap_angle(center + heading_res.reshape(-1))


def decode_camera_geometry(
    encoded_boxes: torch.Tensor,
    depth: torch.Tensor,
    dimensions: torch.Tensor,
    alpha: torch.Tensor,
    projection: torch.Tensor,
    cropped_size: torch.Tensor,
    original_size: torch.Tensor,
    upper: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Decode queries to geometric centers, KITTI dimensions and yaw."""
    count = encoded_boxes.shape[0]
    if count == 0:
        empty = encoded_boxes.new_zeros((0, 3))
        return {"center": empty, "dimensions": empty, "rotation_y": encoded_boxes.new_zeros((0,))}
    if projection.ndim == 2:
        projection = projection.unsqueeze(0).expand(count, -1, -1)
    if cropped_size.ndim == 1:
        cropped_size = cropped_size.unsqueeze(0).expand(count, -1)
    if original_size.ndim == 1:
        original_size = original_size.unsqueeze(0).expand(count, -1)
    upper = upper.reshape(-1)
    if upper.numel() == 1:
        upper = upper.expand(count)

    dtype = encoded_boxes.dtype
    projection = projection.to(dtype=dtype)
    cropped_size = cropped_size.to(dtype=dtype)
    original_size = original_size.to(dtype=dtype)
    upper = upper.to(dtype=dtype)
    depth = depth.reshape(-1).clamp(1.0e-3, 1.0e4)
    dimensions = dimensions.reshape(-1, 3).clamp(1.0e-3, 1.0e3)
    alpha = alpha.reshape(-1)

    crop_height = cropped_size[:, 1].clamp_min(1.0)
    original_width = original_size[:, 0].clamp_min(1.0)
    original_height = original_size[:, 1].clamp_min(1.0)
    vertical_scale = original_height / (crop_height + upper).clamp_min(1.0)
    center_3d_u = encoded_boxes[:, 0] * original_width
    center_3d_v = (encoded_boxes[:, 1] * crop_height + upper) * vertical_scale
    center_2d_x_norm = encoded_boxes[:, 0] + (encoded_boxes[:, 3] - encoded_boxes[:, 2]) * 0.5
    center_2d_u = center_2d_x_norm * original_width

    fu = projection[:, 0, 0].clamp_min(1.0e-6)
    fv = projection[:, 1, 1].clamp_min(1.0e-6)
    cu = projection[:, 0, 2]
    cv = projection[:, 1, 2]
    tx = projection[:, 0, 3] / -fu
    ty = projection[:, 1, 3] / -fv
    x = (center_3d_u - cu) * depth / fu + tx
    y = (center_3d_v - cv) * depth / fv + ty
    rotation_y = wrap_angle(alpha + torch.atan2(center_2d_u - cu, fu))
    return {
        "center": torch.stack((x, y, depth), dim=-1),
        "dimensions": dimensions,
        "rotation_y": rotation_y,
    }


def cuboid_corners(center: torch.Tensor, dimensions: torch.Tensor, rotation_y: torch.Tensor) -> torch.Tensor:
    """Create camera-frame corners from geometric centers."""
    template = center.new_tensor([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ])
    height, width, length = dimensions.unbind(dim=-1)
    scale = torch.stack((length, height, width), dim=-1)
    local = template.unsqueeze(0) * scale.unsqueeze(1)
    cosine = torch.cos(rotation_y)
    sine = torch.sin(rotation_y)
    rotation = center.new_zeros((center.shape[0], 3, 3))
    rotation[:, 0, 0] = cosine
    rotation[:, 0, 2] = sine
    rotation[:, 1, 1] = 1.0
    rotation[:, 2, 0] = -sine
    rotation[:, 2, 2] = cosine
    return torch.bmm(local, rotation.transpose(1, 2)) + center.unsqueeze(1)


def project_corners(corners: torch.Tensor, projection: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    count = corners.shape[0]
    if projection.ndim == 2:
        projection = projection.unsqueeze(0).expand(count, -1, -1)
    homogeneous = torch.cat((corners, torch.ones_like(corners[..., :1])), dim=-1)
    image = torch.bmm(homogeneous, projection.transpose(1, 2))
    z = image[..., 2]
    points = image[..., :2] / z.clamp_min(1.0e-3).unsqueeze(-1)
    return points, z > 1.0e-3


def projected_enclosing_boxes(points: torch.Tensor, original_size: torch.Tensor) -> torch.Tensor:
    if original_size.ndim == 1:
        original_size = original_size.unsqueeze(0).expand(points.shape[0], -1)
    normalized = points / original_size.to(points.dtype).clamp_min(1.0).unsqueeze(1)
    return torch.cat((normalized.amin(dim=1), normalized.amax(dim=1)), dim=-1)


def encoded_boxes_to_original_xyxy(
    encoded_boxes: torch.Tensor,
    cropped_size: torch.Tensor,
    original_size: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    count = encoded_boxes.shape[0]
    if cropped_size.ndim == 1:
        cropped_size = cropped_size.unsqueeze(0).expand(count, -1)
    if original_size.ndim == 1:
        original_size = original_size.unsqueeze(0).expand(count, -1)
    upper = upper.reshape(-1)
    if upper.numel() == 1:
        upper = upper.expand(count)
    crop_height = cropped_size[:, 1].to(encoded_boxes.dtype).clamp_min(1.0)
    original_height = original_size[:, 1].to(encoded_boxes.dtype).clamp_min(1.0)
    upper = upper.to(encoded_boxes.dtype)
    vertical_scale = original_height / (crop_height + upper).clamp_min(1.0)
    center_x = encoded_boxes[:, 0]
    center_y = encoded_boxes[:, 1]
    x_min = center_x - encoded_boxes[:, 2]
    x_max = center_x + encoded_boxes[:, 3]
    y_min = ((center_y - encoded_boxes[:, 4]) * crop_height + upper) * vertical_scale / original_height
    y_max = ((center_y + encoded_boxes[:, 5]) * crop_height + upper) * vertical_scale / original_height
    return torch.stack((x_min, y_min, x_max, y_max), dim=-1)


def paired_generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    right_bottom = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection = (right_bottom - left_top).clamp_min(0.0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0.0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0.0).prod(dim=-1)
    union = (area1 + area2 - intersection).clamp_min(1.0e-8)
    iou = intersection / union
    enclosure_lt = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    enclosure_rb = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    enclosure = (enclosure_rb - enclosure_lt).clamp_min(0.0).prod(dim=-1).clamp_min(1.0e-8)
    return iou - (enclosure - union) / enclosure


def corner_alignment_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_extent = (target.amax(dim=1) - target.amin(dim=1)).norm(dim=-1)
    scale = target_extent.clamp_min(1.0).view(-1, 1, 1)
    return F.smooth_l1_loss(predicted / scale, target / scale, reduction="none").mean(dim=(1, 2))


def projection_alignment_loss(
    predicted_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    boundary_weight: float = 0.25,
) -> torch.Tensor:
    giou = paired_generalized_box_iou(predicted_boxes, target_boxes)
    boundary = F.smooth_l1_loss(predicted_boxes, target_boxes, reduction="none").mean(dim=-1)
    return 1.0 - giou + float(boundary_weight) * boundary


def progressive_weight(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(max((epoch - start_epoch + 1) / float(ramp_epochs), 0.0), 1.0)
