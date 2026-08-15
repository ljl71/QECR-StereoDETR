"""Lightweight, baseline-preserving query depth residual correction.

The module is deliberately much smaller than the detector.  Its final layer is
zero-initialized, so enabling it and loading a V09 checkpoint reproduces V09
depths exactly before the first optimization step.  The geometry-aware variant
uses only the normalized disagreement between StereoDETR's stereo depth and its
already-computed pinhole-geometry depth; it does not add a second backbone or a
dense depth network.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class QueryGeometryDepthResidual(nn.Module):
    """Predict a bounded metric-depth residual for each decoder query."""

    def __init__(
        self,
        query_dim: int,
        hidden_dim: int = 32,
        max_correction: float = 3.0,
        use_geometry_prior: bool = True,
        detach_inputs: bool = True,
    ) -> None:
        super().__init__()
        if query_dim <= 0 or hidden_dim <= 0:
            raise ValueError("query_dim and hidden_dim must be positive")
        if not math.isfinite(max_correction) or max_correction <= 0.0:
            raise ValueError("max_correction must be finite and positive")

        self.max_correction = float(max_correction)
        self.use_geometry_prior = bool(use_geometry_prior)
        self.detach_inputs = bool(detach_inputs)
        # Both ablations allocate exactly the same parameters.  V20A supplies
        # a zero context, while V20B supplies the normalized geometry gap.
        self.layers = nn.Sequential(
            nn.Linear(int(query_dim) + 1, int(hidden_dim)),
            nn.ReLU(inplace=False),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    @staticmethod
    def normalized_geometry_gap(
        stereo_depth: torch.Tensor,
        geometry_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Return a stable, dimensionless geometry-minus-stereo discrepancy."""

        if stereo_depth.shape != geometry_depth.shape:
            raise ValueError(
                "stereo_depth and geometry_depth must have identical shapes"
            )
        denominator = stereo_depth.abs().clamp_min(1.0)
        return ((geometry_depth - stereo_depth) / denominator).clamp(-1.0, 1.0)

    def forward(
        self,
        query_features: torch.Tensor,
        stereo_depth: torch.Tensor,
        geometry_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if stereo_depth.ndim == query_features.ndim - 1:
            stereo_depth = stereo_depth.unsqueeze(-1)
        if geometry_depth.ndim == query_features.ndim - 1:
            geometry_depth = geometry_depth.unsqueeze(-1)
        expected_shape = (*query_features.shape[:-1], 1)
        if stereo_depth.shape != expected_shape:
            raise ValueError(
                "stereo_depth shape {} does not match query shape {}".format(
                    tuple(stereo_depth.shape), tuple(query_features.shape)
                )
            )
        if geometry_depth.shape != expected_shape:
            raise ValueError(
                "geometry_depth shape {} does not match query shape {}".format(
                    tuple(geometry_depth.shape), tuple(query_features.shape)
                )
            )

        features = query_features
        stereo = stereo_depth
        geometry = geometry_depth
        if self.detach_inputs:
            features = features.detach()
            stereo = stereo.detach()
            geometry = geometry.detach()

        if self.use_geometry_prior:
            context = self.normalized_geometry_gap(stereo, geometry)
        else:
            context = torch.zeros_like(stereo)
        raw_residual = self.layers(torch.cat([features, context], dim=-1))
        residual = self.max_correction * torch.tanh(
            raw_residual / self.max_correction
        )
        corrected_depth = (stereo + residual).clamp_min(1.0e-3)
        return {
            "depth": corrected_depth,
            "residual": residual,
            "geometry_context": context,
            "stereo_depth": stereo,
            "geometry_depth": geometry,
        }

