"""Parameter-free depth readouts for StereoDETR depth classification logits."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class ForegroundTopKDepthReadout(nn.Module):
    """Regress depth from the strongest foreground modes.

    StereoDETR reserves the last logit for background.  The original readout
    computes an expectation over all foreground bins plus that background bin.
    This module keeps the original background probability exactly, but
    redistributes the remaining foreground probability mass over only the
    strongest ``top_k`` foreground modes.  Consequently, ``top_k == D``
    recovers the original expectation (up to floating-point round-off), while
    smaller K values cannot turn background pixels into arbitrary near depths.
    """

    def __init__(self, enabled: bool = False, top_k: int = 2) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ValueError("depth_readout.top_k must be positive")

    def forward(
        self,
        depth_logits: torch.Tensor,
        depth_bin_values: torch.Tensor,
        depth_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if depth_logits.ndim != 4:
            raise ValueError("depth_logits must have shape [B, D+1, H, W]")
        if depth_bin_values.ndim != 1:
            raise ValueError("depth_bin_values must be one-dimensional")
        if depth_logits.shape[1] != depth_bin_values.numel():
            raise ValueError("depth logits and bin values disagree")
        if depth_logits.shape[1] < 2:
            raise ValueError("at least one foreground and one background bin are required")

        if depth_probs is None:
            depth_probs = F.softmax(depth_logits, dim=1)
        elif depth_probs.shape != depth_logits.shape:
            raise ValueError("depth_probs must have the same shape as depth_logits")

        bins = depth_bin_values.to(
            device=depth_logits.device,
            dtype=depth_logits.dtype,
        )
        original_depth = (
            depth_probs * bins.reshape(1, -1, 1, 1)
        ).sum(dim=1)
        if not self.enabled:
            return original_depth

        foreground_count = depth_logits.shape[1] - 1
        if self.top_k > foreground_count:
            raise ValueError(
                "depth_readout.top_k={} exceeds {} foreground bins".format(
                    self.top_k,
                    foreground_count,
                )
            )

        foreground_logits = depth_logits[:, :foreground_count]
        selected_logits, selected_indices = torch.topk(
            foreground_logits,
            k=self.top_k,
            dim=1,
            largest=True,
            sorted=False,
        )
        selected_probabilities = F.softmax(selected_logits, dim=1)
        foreground_bins = bins[:foreground_count].reshape(1, -1, 1, 1)
        foreground_bins = foreground_bins.expand_as(foreground_logits)
        selected_bins = torch.gather(
            foreground_bins,
            dim=1,
            index=selected_indices,
        )
        concentrated_foreground_depth = (
            selected_probabilities * selected_bins
        ).sum(dim=1)

        background_probability = depth_probs[:, -1]
        background_depth = bins[-1]
        return (
            (1.0 - background_probability) * concentrated_foreground_depth
            + background_probability * background_depth
        )

