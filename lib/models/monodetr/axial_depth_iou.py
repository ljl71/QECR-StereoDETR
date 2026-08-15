"""Training-only axial IoU objectives for query depth.

The KITTI 3D metric is sensitive to the overlap of an object's occupied
interval along the camera Z axis.  This module supplies a small, parameter-free
1D generalized-IoU objective.  Callers may detach the interval geometry so the
loss updates depth only and does not entangle dimension or heading regression.
"""

from __future__ import annotations

import torch


def axial_half_extent_hwl(
    dimensions_hwl: torch.Tensor,
    rotation_y: torch.Tensor,
    min_half_extent: float = 0.05,
) -> torch.Tensor:
    """Return the Z-axis half extent of rotated KITTI boxes.

    ``dimensions_hwl`` follows the network convention ``[height, width,
    length]``.  KITTI yaw rotates length and width in the camera X-Z plane.
    """

    if dimensions_hwl.shape[-1] != 3:
        raise ValueError("dimensions_hwl must end with [height, width, length]")
    if min_half_extent <= 0.0:
        raise ValueError("min_half_extent must be positive")
    # Match the inference decoder's treatment of dimensions: invalid negative
    # predictions are clamped, never reflected into spuriously large boxes.
    width = dimensions_hwl[..., 1].clamp_min(1.0e-3)
    length = dimensions_hwl[..., 2].clamp_min(1.0e-3)
    half_extent = 0.5 * (
        length * torch.sin(rotation_y).abs()
        + width * torch.cos(rotation_y).abs()
    )
    return half_extent.clamp_min(float(min_half_extent))


def generalized_interval_iou_loss(
    predicted_center: torch.Tensor,
    predicted_half_extent: torch.Tensor,
    target_center: torch.Tensor,
    target_half_extent: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Element-wise 1D generalized-IoU loss.

    Ordinary interval IoU has zero gradient once two intervals no longer
    overlap.  The enclosing-interval penalty in GIoU preserves a useful depth
    gradient in that case.  The returned range is approximately ``[0, 2]``.
    """

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    predicted_center, predicted_half_extent, target_center, target_half_extent = (
        torch.broadcast_tensors(
            predicted_center,
            predicted_half_extent,
            target_center,
            target_half_extent,
        )
    )
    predicted_half_extent = predicted_half_extent.clamp_min(float(eps))
    target_half_extent = target_half_extent.clamp_min(float(eps))

    predicted_left = predicted_center - predicted_half_extent
    predicted_right = predicted_center + predicted_half_extent
    target_left = target_center - target_half_extent
    target_right = target_center + target_half_extent

    intersection = (
        torch.minimum(predicted_right, target_right)
        - torch.maximum(predicted_left, target_left)
    ).clamp_min(0.0)
    predicted_length = 2.0 * predicted_half_extent
    target_length = 2.0 * target_half_extent
    union = (predicted_length + target_length - intersection).clamp_min(eps)
    interval_iou = intersection / union

    enclosing = (
        torch.maximum(predicted_right, target_right)
        - torch.minimum(predicted_left, target_left)
    ).clamp_min(eps)
    generalized_iou = interval_iou - (enclosing - union) / enclosing
    return 1.0 - generalized_iou


def depth_only_axial_giou_loss(
    predicted_depth: torch.Tensor,
    target_depth: torch.Tensor,
    predicted_dimensions_hwl: torch.Tensor,
    target_dimensions_hwl: torch.Tensor,
    predicted_rotation_y: torch.Tensor,
    target_rotation_y: torch.Tensor,
    min_half_extent: float = 0.05,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Element-wise axial GIoU with geometry detached from the gradient.

    Detaching both half extents is deliberate: V22 is a depth objective, not a
    second dimension or heading loss.  The original StereoDETR losses continue
    to supervise those attributes independently.
    """

    predicted_half_extent = axial_half_extent_hwl(
        predicted_dimensions_hwl.detach(),
        predicted_rotation_y.detach(),
        min_half_extent=min_half_extent,
    ).detach()
    target_half_extent = axial_half_extent_hwl(
        target_dimensions_hwl.detach(),
        target_rotation_y.detach(),
        min_half_extent=min_half_extent,
    ).detach()
    return generalized_interval_iou_loss(
        predicted_depth,
        predicted_half_extent,
        target_depth.detach(),
        target_half_extent,
        eps=eps,
    )
