"""Runtime-tested sparse query epipolar refinement.

This is the execution module used by the QECR bootstrap.  It is kept separate
because the Windows workspace cannot overwrite an already-created source file
after a test-driven fix.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def _normalized_entropy(probabilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    count = probabilities.shape[-1]
    entropy = -(
        probabilities * probabilities.clamp_min(eps).log()
    ).sum(dim=-1, keepdim=True)
    if count > 1:
        entropy = entropy / probabilities.new_tensor(float(count)).log()
    return entropy


class QueryEpipolarConsistencyRefiner(nn.Module):
    """Sparse left/right matching and local query-depth posterior refinement."""

    def __init__(
        self,
        num_samples: int = 5,
        search_radius: float = 2.0,
        matching_temperature: float = 0.07,
        depth_temperature: float = 1.0,
        fusion_weight: float = 0.5,
        bidirectional: bool = True,
        cycle_threshold: float = 1.0,
        detach_points: bool = True,
        detach_depth_center: bool = True,
        align_corners: bool = True,
        min_disparity_pixels: float = 0.25,
    ) -> None:
        super().__init__()
        if num_samples < 3 or num_samples % 2 == 0:
            raise ValueError("num_samples must be an odd integer >= 3")
        if search_radius <= 0:
            raise ValueError("search_radius must be positive")
        if matching_temperature <= 0 or depth_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if fusion_weight < 0:
            raise ValueError("fusion_weight must be non-negative")
        if cycle_threshold <= 0:
            raise ValueError("cycle_threshold must be positive")

        self.num_samples = int(num_samples)
        self.search_radius = float(search_radius)
        self.matching_temperature = float(matching_temperature)
        self.depth_temperature = float(depth_temperature)
        self.fusion_weight = float(fusion_weight)
        self.bidirectional = bool(bidirectional)
        self.cycle_threshold = float(cycle_threshold)
        self.detach_points = bool(detach_points)
        self.detach_depth_center = bool(detach_depth_center)
        self.align_corners = bool(align_corners)
        self.min_disparity_pixels = float(min_disparity_pixels)
        self.register_buffer(
            "sample_offsets",
            torch.linspace(
                -self.search_radius,
                self.search_radius,
                steps=self.num_samples,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def _sample(
        self,
        feature_map: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        sampled = F.grid_sample(
            feature_map,
            points.mul(2.0).sub(1.0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )
        return sampled.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _valid_points(points: torch.Tensor) -> torch.Tensor:
        return (
            (points[..., 0] >= 0.0)
            & (points[..., 0] <= 1.0)
            & (points[..., 1] >= 0.0)
            & (points[..., 1] <= 1.0)
        )

    def _masked_probabilities(
        self,
        logits: torch.Tensor,
        valid: torch.Tensor,
    ):
        probabilities = F.softmax(
            logits.masked_fill(~valid, -1.0e4)
            / self.matching_temperature,
            dim=-1,
        )
        has_valid = valid.any(dim=-1, keepdim=True)
        probabilities = torch.where(
            has_valid,
            probabilities,
            torch.full_like(probabilities, 1.0 / self.num_samples),
        )
        confidence = (
            1.0 - _normalized_entropy(probabilities)
        ) * has_valid.to(probabilities.dtype)
        return probabilities, confidence, has_valid

    def forward(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        query_points: torch.Tensor,
        local_logits: torch.Tensor,
        local_bins: torch.Tensor,
        local_indices: torch.Tensor,
        full_probabilities: torch.Tensor,
        stereo_fb: torch.Tensor,
        image_width: torch.Tensor,
        stereo_direction: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if left_features.shape != right_features.shape:
            raise ValueError("left and right feature maps must have equal shapes")
        if left_features.ndim != 4:
            raise ValueError("feature maps must have shape [B, C, H, W]")
        if query_points.ndim != 3 or query_points.shape[-1] != 2:
            raise ValueError("query_points must have shape [B, Q, 2]")
        if local_logits.shape != local_bins.shape:
            raise ValueError("local_logits and local_bins must have equal shapes")
        if local_indices.shape != local_logits.shape:
            raise ValueError("local_indices and local_logits must have equal shapes")

        batch_size, _, _, feature_width = left_features.shape
        if feature_width < 2:
            raise ValueError("epipolar matching requires feature width >= 2")

        points = query_points.detach() if self.detach_points else query_points
        base_probabilities = F.softmax(local_logits, dim=-1)
        initial_depth = (
            base_probabilities * local_bins
        ).sum(dim=-1, keepdim=True).clamp_min(1.0e-3)
        if self.detach_depth_center:
            initial_depth = initial_depth.detach()

        fb = stereo_fb.to(
            device=left_features.device,
            dtype=left_features.dtype,
        ).reshape(batch_size, 1, 1).clamp_min(1.0e-6)
        width = image_width.to(
            device=left_features.device,
            dtype=left_features.dtype,
        ).reshape(batch_size, 1, 1).clamp_min(2.0)
        direction = stereo_direction.to(
            device=left_features.device,
            dtype=left_features.dtype,
        ).reshape(batch_size, 1, 1)
        if not torch.all((direction == -1) | (direction == 1)):
            raise ValueError("stereo_direction values must be either -1 or +1")

        disparity_pixels = fb / initial_depth
        disparity_normalized = disparity_pixels / width
        right_center = points[..., 0:1] + direction * disparity_normalized

        offsets = self.sample_offsets.to(
            device=left_features.device,
            dtype=left_features.dtype,
        ).view(1, 1, -1)
        offsets_normalized = offsets / float(feature_width - 1)
        right_points = torch.stack(
            [
                right_center + offsets_normalized,
                points[..., 1:2].expand(-1, -1, self.num_samples),
            ],
            dim=-1,
        )
        right_valid = self._valid_points(right_points)

        left_query = self._sample(
            left_features,
            points.unsqueeze(2),
        ).squeeze(2)
        right_candidates = self._sample(right_features, right_points)
        forward_logits = (
            F.normalize(right_candidates, dim=-1, eps=1.0e-6)
            * F.normalize(left_query, dim=-1, eps=1.0e-6).unsqueeze(2)
        ).sum(dim=-1)
        forward_probabilities, forward_confidence, has_right = (
            self._masked_probabilities(forward_logits, right_valid)
        )
        forward_offset = (
            forward_probabilities * offsets_normalized
        ).sum(dim=-1, keepdim=True)
        refined_right_x = right_center + forward_offset
        refined_disparity_pixels = (
            refined_right_x - points[..., 0:1]
        ).abs() * width
        valid_disparity = (
            refined_disparity_pixels >= self.min_disparity_pixels
        )
        epipolar_depth = (
            fb / refined_disparity_pixels.clamp_min(
                self.min_disparity_pixels
            )
        )
        max_local_depth = local_bins.detach().amax(dim=-1, keepdim=True)
        epipolar_depth = torch.minimum(
            epipolar_depth.clamp_min(1.0e-3),
            max_local_depth + 20.0,
        )

        cycle_error = torch.zeros_like(epipolar_depth)
        reverse_confidence = torch.ones_like(forward_confidence)
        has_reverse = torch.ones_like(has_right)
        if self.bidirectional:
            right_query = (
                forward_probabilities.unsqueeze(-1) * right_candidates
            ).sum(dim=2)
            reverse_center = (
                refined_right_x - direction * disparity_normalized
            )
            left_points = torch.stack(
                [
                    reverse_center + offsets_normalized,
                    points[..., 1:2].expand(-1, -1, self.num_samples),
                ],
                dim=-1,
            )
            left_valid = self._valid_points(left_points)
            left_candidates = self._sample(left_features, left_points)
            reverse_logits = (
                F.normalize(left_candidates, dim=-1, eps=1.0e-6)
                * F.normalize(right_query, dim=-1, eps=1.0e-6).unsqueeze(2)
            ).sum(dim=-1)
            reverse_probabilities, reverse_confidence, has_reverse = (
                self._masked_probabilities(reverse_logits, left_valid)
            )
            reverse_offset = (
                reverse_probabilities * offsets_normalized
            ).sum(dim=-1, keepdim=True)
            returned_left_x = reverse_center + reverse_offset
            cycle_error = (
                returned_left_x - points[..., 0:1]
            ).abs() * float(feature_width - 1)

        threshold = cycle_error.new_tensor(self.cycle_threshold)
        cycle_confidence = (
            F.softplus(threshold - cycle_error)
            / F.softplus(threshold)
        ).clamp(0.0, 1.0)
        valid = has_right & has_reverse & valid_disparity
        confidence = (
            forward_confidence
            * reverse_confidence
            * cycle_confidence
            * valid.to(forward_confidence.dtype)
        )

        epipolar_logits = -(
            local_bins - epipolar_depth
        ).abs() / self.depth_temperature
        fused_logits = (
            local_logits
            + self.fusion_weight * confidence * epipolar_logits
        )
        fused_probabilities = F.softmax(fused_logits, dim=-1)
        expected_depth = (
            fused_probabilities * local_bins
        ).sum(dim=-1, keepdim=True)
        entropy = _normalized_entropy(fused_probabilities)

        fused_full_probabilities = full_probabilities.clone()
        fused_full_probabilities.scatter_(
            -1,
            local_indices,
            fused_probabilities,
        )
        fused_full_probabilities = (
            fused_full_probabilities
            / fused_full_probabilities.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
        )
        return {
            "logits": fused_logits,
            "probabilities": fused_probabilities,
            "full_probabilities": fused_full_probabilities,
            "expected_depth": expected_depth,
            "entropy": entropy,
            "epipolar_depth": epipolar_depth,
            "confidence": confidence,
            "cycle_error": cycle_error,
            "valid": valid.to(fused_probabilities.dtype),
            "forward_confidence": forward_confidence,
            "reverse_confidence": reverse_confidence,
            "refined_disparity_pixels": refined_disparity_pixels,
        }

