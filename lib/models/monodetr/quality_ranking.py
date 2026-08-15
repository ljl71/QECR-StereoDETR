"""Lightweight query-level 3D quality ranking for StereoDETR.

The head follows the official RARE ranking-head layout.  Ground-truth quality
is computed only while training, from the detector's decoded camera-frame 3D
boxes and KITTI 3D IoU.  No IoU calculation is present in inference.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Query3DQualityHead(nn.Module):
    """RARE-style two-layer quality head applied to decoder queries."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        for module in self.layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        if query_features.ndim != 3:
            raise ValueError("query features must have shape [B, Q, C]")
        batch, queries, channels = query_features.shape
        logits = self.layers(query_features.reshape(-1, channels))
        return logits.reshape(batch, queries, 1)


def quality_probability_from_logits(
    quality_logits: torch.Tensor,
    score_power: float = 1.0,
) -> torch.Tensor:
    """Calibrate inference quality without changing the trained head.

    ``score_power=0`` removes the quality factor, ``1`` reproduces V06B/V06C,
    and fractional values soften its influence.  This operation has no
    trainable state, so the same checkpoint can be evaluated at every power.
    """

    score_power = float(score_power)
    if not math.isfinite(score_power) or score_power < 0.0:
        raise ValueError("quality score_power must be finite and non-negative")
    return quality_logits.sigmoid().pow(score_power)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _decode_heading(heading: np.ndarray) -> np.ndarray:
    bins = np.argmax(heading[..., :12], axis=-1)
    residuals = np.take_along_axis(
        heading[..., 12:24], bins[..., None], axis=-1
    )[..., 0]
    return _wrap_angle(bins.astype(np.float64) * (2.0 * np.pi / 12.0) + residuals)


def _decode_target_heading(target: Dict[str, torch.Tensor]) -> np.ndarray:
    bins = target["heading_bin"].detach().cpu().numpy().reshape(-1)
    residuals = target["heading_res"].detach().cpu().numpy().reshape(-1)
    return _wrap_angle(bins.astype(np.float64) * (2.0 * np.pi / 12.0) + residuals)


def _camera_boxes(
    encoded_boxes: np.ndarray,
    depth: np.ndarray,
    dimensions: np.ndarray,
    alpha: np.ndarray,
    projection: np.ndarray,
    cropped_size: np.ndarray,
    original_size: np.ndarray,
    upper: float,
) -> np.ndarray:
    """Decode StereoDETR box tensors to KITTI [x,y,z,h,w,l,ry]."""

    encoded_boxes = np.asarray(encoded_boxes, dtype=np.float64)
    count = encoded_boxes.shape[0]
    if count == 0:
        return np.empty((0, 7), dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    dimensions = np.asarray(dimensions, dtype=np.float64).reshape(-1, 3)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    depth = np.clip(depth, 1.0e-3, 1.0e4)
    dimensions = np.clip(dimensions, 1.0e-3, 1.0e3)

    crop_width, crop_height = np.asarray(cropped_size, dtype=np.float64)
    original_width, original_height = np.asarray(original_size, dtype=np.float64)
    vertical_scale = original_height / max(crop_height + float(upper), 1.0)

    center_3d_u = encoded_boxes[:, 0] * original_width
    center_3d_v = (
        encoded_boxes[:, 1] * crop_height + float(upper)
    ) * vertical_scale
    center_2d_x_norm = encoded_boxes[:, 0] + (
        encoded_boxes[:, 3] - encoded_boxes[:, 2]
    ) * 0.5
    center_2d_u = center_2d_x_norm * original_width

    fu = float(projection[0, 0])
    fv = float(projection[1, 1])
    cu = float(projection[0, 2])
    cv = float(projection[1, 2])
    tx = float(projection[0, 3]) / -fu
    ty = float(projection[1, 3]) / -fv
    x = (center_3d_u - cu) * depth / fu + tx
    y = (center_3d_v - cv) * depth / fv + ty
    y = y + dimensions[:, 0] * 0.5
    rotation_y = _wrap_angle(alpha + np.arctan2(center_2d_u - cu, fu))
    return np.column_stack(
        [x, y, depth, dimensions[:, 0], dimensions[:, 1], dimensions[:, 2], rotation_y]
    )


def _assert_canonical_training_geometry(targets: Sequence[Dict[str, torch.Tensor]]) -> None:
    """Reject augmented geometry that cannot be decoded by the canonical path."""

    for target in targets:
        for key in ("random_flip_flag", "random_switch_flag", "random_crop_flag", "random_mix_flag"):
            if key in target and bool(torch.as_tensor(target[key]).reshape(-1)[0].item()):
                raise ValueError(
                    "3D quality supervision requires geometric augmentation to be disabled; "
                    f"observed {key}=True"
                )
        if "crop_scale" in target:
            scale = float(torch.as_tensor(target["crop_scale"]).reshape(-1)[0].item())
            if abs(scale - 1.0) > 1.0e-6:
                raise ValueError(
                    "3D quality supervision requires crop_scale=1; observed {:.6f}".format(scale)
                )


@torch.no_grad()
def build_3d_iou_quality_targets(
    outputs: Dict[str, torch.Tensor],
    targets: Sequence[Dict[str, torch.Tensor]],
    overlap_fn: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None,
) -> torch.Tensor:
    """Return each query's maximum 3D IoU to a same-class ground truth."""

    required = {
        "pred_logits",
        "pred_boxes",
        "pred_depth",
        "pred_3d_dim",
        "pred_angle",
        "quality_calibs",
        "quality_img_sizes",
        "quality_img_sizes_ori",
        "quality_img_sizes_upper",
    }
    missing = required.difference(outputs)
    if missing:
        raise KeyError("quality target metadata missing: {}".format(sorted(missing)))
    _assert_canonical_training_geometry(targets)
    if overlap_fn is None:
        from lib.datasets.kitti.kitti_eval_python.eval import d3_box_overlap

        overlap_fn = d3_box_overlap

    logits = outputs["pred_logits"].detach()
    device = logits.device
    dtype = logits.dtype
    batch_size, query_count = logits.shape[:2]
    predicted_labels = logits.sigmoid().argmax(dim=-1).cpu().numpy()
    predicted_boxes = outputs["pred_boxes"].detach().cpu().numpy()
    predicted_depth = outputs["pred_depth"][..., 0].detach().cpu().numpy()
    predicted_dims = outputs["pred_3d_dim"].detach().cpu().numpy()
    predicted_heading = outputs["pred_angle"].detach().cpu().numpy()
    calibrations = outputs["quality_calibs"].detach().cpu().numpy()
    cropped_sizes = outputs["quality_img_sizes"].detach().cpu().numpy()
    original_sizes = outputs["quality_img_sizes_ori"].detach().cpu().numpy()
    upper_values = outputs["quality_img_sizes_upper"].detach().cpu().numpy()

    all_quality: List[np.ndarray] = []
    for batch_index in range(batch_size):
        projection = calibrations[batch_index]
        pred_camera_boxes = _camera_boxes(
            predicted_boxes[batch_index],
            predicted_depth[batch_index],
            predicted_dims[batch_index],
            _decode_heading(predicted_heading[batch_index]),
            projection,
            cropped_sizes[batch_index],
            original_sizes[batch_index],
            float(np.asarray(upper_values[batch_index]).reshape(-1)[0]),
        )
        target = targets[batch_index]
        gt_labels = target["labels"].detach().cpu().numpy().reshape(-1)
        gt_camera_boxes = _camera_boxes(
            target["boxes_3d"].detach().cpu().numpy(),
            target["depth"].detach().cpu().numpy(),
            target["size_3d"].detach().cpu().numpy(),
            _decode_target_heading(target),
            projection,
            cropped_sizes[batch_index],
            original_sizes[batch_index],
            float(np.asarray(upper_values[batch_index]).reshape(-1)[0]),
        )
        quality = np.zeros(query_count, dtype=np.float32)
        if len(gt_labels):
            overlaps = np.asarray(
                overlap_fn(pred_camera_boxes, gt_camera_boxes),
                dtype=np.float64,
            )
            same_class = (
                predicted_labels[batch_index, :, None]
                == gt_labels[None, :]
            )
            overlaps = np.where(same_class, overlaps, 0.0)
            quality = np.max(overlaps, axis=1).astype(np.float32)
        all_quality.append(np.clip(quality, 0.0, 1.0))
    return torch.as_tensor(np.stack(all_quality), device=device, dtype=dtype)


def pointwise_quality_loss(
    quality_logits: torch.Tensor,
    quality_targets: torch.Tensor,
    negative_threshold: float = 0.1,
    negative_weight: float = 0.1,
) -> torch.Tensor:
    quality = quality_logits.squeeze(-1).sigmoid()
    weights = torch.ones_like(quality_targets)
    weights = torch.where(
        quality_targets < float(negative_threshold),
        weights * float(negative_weight),
        weights,
    )
    return (weights * (quality - quality_targets).square()).sum() / weights.sum().clamp_min(1.0)


def pairwise_quality_loss(
    quality_logits: torch.Tensor,
    quality_targets: torch.Tensor,
    predicted_labels: torch.Tensor,
    margin: float = 0.1,
    max_pairs_per_class: int = 32,
) -> torch.Tensor:
    """RARE-style class-wise pair ranking on quality-logit differences."""

    logits = quality_logits.squeeze(-1)
    losses: List[torch.Tensor] = []
    for batch_index in range(logits.shape[0]):
        for class_index in torch.unique(predicted_labels[batch_index]):
            indices = torch.nonzero(
                predicted_labels[batch_index] == class_index,
                as_tuple=False,
            ).flatten()
            if indices.numel() < 2:
                continue
            target = quality_targets[batch_index, indices]
            differences = target[:, None] - target[None, :]
            valid = torch.triu(differences.abs() >= float(margin), diagonal=1)
            pairs = torch.nonzero(valid, as_tuple=False)
            if pairs.numel() == 0:
                continue
            if pairs.shape[0] > int(max_pairs_per_class):
                order = torch.randperm(pairs.shape[0], device=pairs.device)[
                    : int(max_pairs_per_class)
                ]
                pairs = pairs[order]
            first = indices[pairs[:, 0]]
            second = indices[pairs[:, 1]]
            ranking_target = (
                quality_targets[batch_index, first]
                > quality_targets[batch_index, second]
            ).to(logits.dtype)
            losses.append(
                F.binary_cross_entropy_with_logits(
                    logits[batch_index, first] - logits[batch_index, second],
                    ranking_target,
                )
            )
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()
