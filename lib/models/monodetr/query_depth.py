"""Lightweight query-level probabilistic depth modules for QDR-StereoDETR.

The modules in this file operate only on object queries. They intentionally
reuse StereoDETR's existing dense depth logits instead of adding another dense
stereo branch.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


def _normalized_entropy(probabilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return entropy in [0, 1] along the last dimension."""
    num_bins = probabilities.shape[-1]
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1, keepdim=True)
    if num_bins > 1:
        entropy = entropy / probabilities.new_tensor(float(num_bins)).log()
    return entropy


class QueryDepthDistributionSampler(nn.Module):
    """Sample dense depth logits at one or a few locations per object query."""

    def __init__(
        self,
        hidden_dim: int,
        num_decoder_layers: int,
        num_points: int = 1,
        temperature: float = 1.0,
        point_radius: float = 0.25,
        entropy_weight: float = 1.0,
        background_weight: float = 1.0,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        if num_points < 1:
            raise ValueError("num_points must be at least one")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.num_points = int(num_points)
        self.temperature = float(temperature)
        self.point_radius = float(point_radius)
        self.entropy_weight = float(entropy_weight)
        self.background_weight = float(background_weight)
        self.align_corners = bool(align_corners)

        if self.num_points > 1:
            self.offset_heads = nn.ModuleList(
                nn.Linear(hidden_dim, 2 * (self.num_points - 1))
                for _ in range(num_decoder_layers)
            )
            self.point_score_heads = nn.ModuleList(
                nn.Linear(hidden_dim, self.num_points)
                for _ in range(num_decoder_layers)
            )
            for head in self.offset_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            for head in self.point_score_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

            base_offsets = self._make_base_offsets(self.num_points - 1)
            self.register_buffer("base_offsets", base_offsets, persistent=False)
        else:
            self.offset_heads = None
            self.point_score_heads = None
            self.register_buffer("base_offsets", torch.zeros(0, 2), persistent=False)

    @staticmethod
    def _make_base_offsets(num_offsets: int) -> torch.Tensor:
        """Deterministic cross/diagonal pattern used before learned residuals."""
        pattern = torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [0.7071, 0.7071],
                [-0.7071, 0.7071],
                [0.7071, -0.7071],
                [-0.7071, -0.7071],
            ],
            dtype=torch.float32,
        )
        repeats = (num_offsets + pattern.shape[0] - 1) // pattern.shape[0]
        return pattern.repeat(repeats, 1)[:num_offsets]

    def _build_points(
        self,
        query_features: torch.Tensor,
        center_points: torch.Tensor,
        box_extents: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        center_points = center_points[..., :2]
        if self.num_points == 1:
            return center_points.unsqueeze(2)

        batch_size, num_queries, _ = query_features.shape
        learned = self.offset_heads[layer_index](query_features)
        learned = learned.view(batch_size, num_queries, self.num_points - 1, 2)
        learned = 0.5 * torch.tanh(learned)
        base = self.base_offsets.to(dtype=query_features.dtype).view(
            1, 1, self.num_points - 1, 2
        )
        extent = box_extents.clamp_min(1e-4).unsqueeze(2)
        offsets = (base + learned) * extent * self.point_radius
        extra_points = center_points.unsqueeze(2) + offsets
        points = torch.cat([center_points.unsqueeze(2), extra_points], dim=2)
        return points.clamp(0.0, 1.0)

    def forward(
        self,
        depth_logits: torch.Tensor,
        depth_bin_values: torch.Tensor,
        query_features: torch.Tensor,
        center_points: torch.Tensor,
        box_extents: torch.Tensor,
        layer_index: int,
    ) -> Dict[str, torch.Tensor]:
        if depth_logits.ndim != 4:
            raise ValueError("depth_logits must have shape [B, D+1, H, W]")

        valid_bin_count = depth_bin_values.numel() - 1
        if valid_bin_count < 2:
            raise ValueError("depth_bin_values must contain valid bins plus a background bin")
        if depth_logits.shape[1] != depth_bin_values.numel():
            raise ValueError("depth logits and depth-bin values have incompatible channel counts")

        points = self._build_points(
            query_features, center_points, box_extents, layer_index
        )
        sampling_grid = points.mul(2.0).sub(1.0)
        sampled_logits = F.grid_sample(
            depth_logits,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=self.align_corners,
        )
        sampled_logits = sampled_logits.permute(0, 2, 3, 1).contiguous()

        all_probabilities = F.softmax(sampled_logits / self.temperature, dim=-1)
        background_probability = all_probabilities[..., valid_bin_count:]
        valid_logits = sampled_logits[..., :valid_bin_count]
        point_probabilities = F.softmax(valid_logits / self.temperature, dim=-1)
        point_entropy = _normalized_entropy(point_probabilities)

        if self.num_points == 1:
            point_weights = torch.ones_like(point_entropy)
        else:
            learned_scores = self.point_score_heads[layer_index](query_features).unsqueeze(-1)
            reliability = (
                learned_scores
                - self.entropy_weight * point_entropy
                - self.background_weight * background_probability
            )
            point_weights = F.softmax(reliability, dim=2)

        probabilities = (point_weights * point_probabilities).sum(dim=2)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = _normalized_entropy(probabilities)
        background_probability = (point_weights * background_probability).sum(dim=2)

        valid_bins = depth_bin_values[:valid_bin_count].to(
            device=depth_logits.device, dtype=depth_logits.dtype
        )
        point_depth = (point_probabilities * valid_bins.view(1, 1, 1, -1)).sum(
            dim=-1, keepdim=True
        )
        expected_depth = (probabilities * valid_bins.view(1, 1, -1)).sum(
            dim=-1, keepdim=True
        )
        point_variance = (
            point_weights * (point_depth - expected_depth.unsqueeze(2)).pow(2)
        ).sum(dim=2)

        return {
            "points": points,
            "point_weights": point_weights,
            "point_probabilities": point_probabilities,
            "probabilities": probabilities,
            "logits": probabilities.clamp_min(1e-8).log(),
            "expected_depth": expected_depth,
            "entropy": entropy,
            "background_probability": background_probability,
            "point_depth_variance": point_variance,
        }


class BaselineSafeDepthFusion(nn.Module):
    """Blend query depth into StereoDETR depth without replacing the baseline.

    Each decoder layer owns one bounded scalar query weight.  The scalar starts
    close to zero, so a newly initialized model behaves mostly like the
    original StereoDETR depth path while the query-distribution branch learns.
    """

    def __init__(
        self,
        num_decoder_layers: int,
        initial_query_weight: float = 0.05,
        max_query_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if num_decoder_layers < 1:
            raise ValueError("num_decoder_layers must be at least one")
        if not 0.0 < max_query_weight <= 1.0:
            raise ValueError("max_query_weight must be in (0, 1]")
        if not 0.0 < initial_query_weight < max_query_weight:
            raise ValueError(
                "initial_query_weight must be in (0, max_query_weight)"
            )

        self.max_query_weight = float(max_query_weight)
        initial_ratio = initial_query_weight / self.max_query_weight
        initial_logit = math.log(initial_ratio / (1.0 - initial_ratio))
        self.query_weight_logits = nn.Parameter(
            torch.full((num_decoder_layers,), initial_logit)
        )

    def forward(
        self,
        baseline_depth: torch.Tensor,
        query_depth: torch.Tensor,
        layer_index: int,
    ) -> Dict[str, torch.Tensor]:
        if baseline_depth.shape != query_depth.shape:
            raise ValueError(
                "baseline_depth and query_depth must have identical shapes"
            )
        if not 0 <= layer_index < self.query_weight_logits.numel():
            raise IndexError("layer_index is out of range")

        scalar_weight = self.max_query_weight * torch.sigmoid(
            self.query_weight_logits[layer_index]
        )
        query_weight = scalar_weight.reshape(1, 1, 1).expand_as(
            baseline_depth
        )
        residual = query_depth - baseline_depth
        fused_depth = baseline_depth + query_weight * residual
        return {
            "depth": fused_depth,
            "query_weight": query_weight,
            "baseline_depth": baseline_depth,
            "query_depth": query_depth,
            "residual": residual,
        }


class QueryDepthDistributionRefiner(nn.Module):
    """D-FINE-inspired residual refinement on a compact local depth support."""

    def __init__(
        self,
        hidden_dim: int,
        num_decoder_layers: int,
        total_bins: int,
        local_bins: int = 16,
        enabled: bool = False,
        prior_weight: float = 0.5,
        detach_window: bool = True,
        posterior_floor: float = 0.05,
    ) -> None:
        super().__init__()
        if local_bins < 2 or local_bins > total_bins:
            raise ValueError("local_bins must be in [2, total_bins]")

        self.total_bins = int(total_bins)
        self.local_bins = int(local_bins)
        self.enabled = bool(enabled)
        self.prior_weight = float(prior_weight)
        self.detach_window = bool(detach_window)
        self.posterior_floor = float(posterior_floor)

        if self.enabled:
            self.residual_heads = nn.ModuleList(
                nn.Linear(hidden_dim, self.local_bins)
                for _ in range(num_decoder_layers)
            )
            for head in self.residual_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        else:
            self.residual_heads = None

    def _local_indices(self, probabilities: torch.Tensor) -> torch.Tensor:
        if self.local_bins == self.total_bins:
            indices = torch.arange(
                self.total_bins, device=probabilities.device, dtype=torch.long
            )
            return indices.view(1, 1, -1).expand(
                probabilities.shape[0], probabilities.shape[1], -1
            )

        index_values = torch.arange(
            self.total_bins,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        center = (probabilities * index_values.view(1, 1, -1)).sum(dim=-1)
        if self.detach_window:
            center = center.detach()
        center = center.round().long()

        half = self.local_bins // 2
        start = (center - half).clamp(0, self.total_bins - self.local_bins)
        offsets = torch.arange(
            self.local_bins, device=probabilities.device, dtype=torch.long
        )
        return start.unsqueeze(-1) + offsets.view(1, 1, -1)

    def forward(
        self,
        observed_probabilities: torch.Tensor,
        depth_bin_values: torch.Tensor,
        query_features: torch.Tensor,
        layer_index: int,
        previous_full_probabilities: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        observed_probabilities = observed_probabilities.clamp_min(1e-8)
        if self.enabled and previous_full_probabilities is not None:
            previous = previous_full_probabilities.clamp_min(1e-8)
            base_full = observed_probabilities * previous.pow(self.prior_weight)
            base_full = base_full / base_full.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            base_full = observed_probabilities

        indices = self._local_indices(base_full)
        local_prior = base_full.gather(-1, indices).clamp_min(1e-8)
        local_prior = local_prior / local_prior.sum(dim=-1, keepdim=True)

        if self.enabled:
            residual_logits = self.residual_heads[layer_index](query_features)
        else:
            residual_logits = torch.zeros_like(local_prior)

        local_logits = local_prior.log() + residual_logits
        local_probabilities = F.softmax(local_logits, dim=-1)

        valid_bins = depth_bin_values[: self.total_bins].to(
            device=observed_probabilities.device,
            dtype=observed_probabilities.dtype,
        )
        expanded_bins = valid_bins.view(1, 1, -1).expand_as(base_full)
        local_depth_bins = expanded_bins.gather(-1, indices)
        expected_depth = (local_probabilities * local_depth_bins).sum(
            dim=-1, keepdim=True
        )
        entropy = _normalized_entropy(local_probabilities)

        # Keep a small observed-probability floor outside the local window so
        # later decoder layers can recover from an initially misplaced window.
        full_probabilities = base_full * self.posterior_floor
        full_probabilities.scatter_(-1, indices, local_probabilities)
        full_probabilities = full_probabilities / full_probabilities.sum(
            dim=-1, keepdim=True
        )

        return {
            "indices": indices,
            "bins": local_depth_bins,
            "logits": local_logits,
            "probabilities": local_probabilities,
            "full_probabilities": full_probabilities,
            "expected_depth": expected_depth,
            "entropy": entropy,
        }


class UncertaintyGeometryGate(nn.Module):
    """Fuse stereo and projective geometry depth with a guarded residual gate."""

    def __init__(
        self,
        hidden_dim: int,
        hidden_gate_dim: int = 32,
        initial_stereo_weight: float = 0.9,
        max_correction: float = 10.0,
    ) -> None:
        super().__init__()
        self.max_correction = float(max_correction)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_gate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_gate_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        initial_stereo_weight = min(max(initial_stereo_weight, 1e-4), 1.0 - 1e-4)
        initial_bias = torch.log(
            torch.tensor(initial_stereo_weight / (1.0 - initial_stereo_weight))
        )
        nn.init.constant_(self.gate[-1].bias, float(initial_bias))

    def forward(
        self,
        query_features: torch.Tensor,
        stereo_depth: torch.Tensor,
        geometry_depth: torch.Tensor,
        entropy: torch.Tensor,
        background_probability: torch.Tensor,
        log_variance: torch.Tensor,
        point_depth_variance: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        stereo_depth = stereo_depth.clamp_min(1e-3)
        geometry_depth = geometry_depth.clamp_min(1e-3)
        relative_disagreement = (
            (stereo_depth - geometry_depth).abs() / stereo_depth.detach().clamp_min(1e-3)
        ).clamp(max=5.0)
        uncertainty_features = torch.cat(
            [
                entropy,
                background_probability,
                relative_disagreement,
                log_variance.clamp(-5.0, 5.0),
                point_depth_variance.sqrt().clamp(max=20.0),
            ],
            dim=-1,
        )
        stereo_weight = torch.sigmoid(
            self.gate(torch.cat([query_features, uncertainty_features], dim=-1))
        )
        correction = (geometry_depth - stereo_depth).clamp(
            min=-self.max_correction, max=self.max_correction
        )
        fused_depth = stereo_depth + (1.0 - stereo_weight) * correction
        return {
            "depth": fused_depth,
            "stereo_weight": stereo_weight,
            "geometry_depth": geometry_depth,
            "stereo_depth": stereo_depth,
        }
