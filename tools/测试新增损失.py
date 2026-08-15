"""Finite-value and backward checks for the added criterion losses."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn.modules.linear as torch_linear
import torch.overrides as torch_overrides

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Compatibility shim used only by this test: the official repository imports
# a private PyTorch alias removed from newer releases.
if not hasattr(torch_linear, "_LinearWithBias"):
    torch_linear._LinearWithBias = torch.nn.Linear
sys.modules.setdefault("torch._overrides", torch_overrides)

# The criterion itself does not execute deformable attention, but importing the
# official model package normally requires its compiled CUDA extension.
fake_msda = types.ModuleType("MultiScaleDeformableAttention")
fake_msda.ms_deform_attn_forward = lambda *args, **kwargs: None
fake_msda.ms_deform_attn_backward = lambda *args, **kwargs: None
sys.modules.setdefault("MultiScaleDeformableAttention", fake_msda)

from lib.models.monodetr.stereodetr import SetCriterion  # noqa: E402


def main() -> None:
    torch.manual_seed(13)
    batch_size, num_queries, num_classes = 2, 5, 4
    local_bins, num_points = 8, 5
    query_config = {
        "quality_aligned_cls": True,
        "quality_iou_alpha": 0.5,
        "quality_depth_tau": 0.1,
        "target_temperature": 1.0,
        "gate_oracle_temperature": 1.0,
        "teacher_confidence_threshold": 0.2,
        "teacher_target_temperature": 1.0,
        "align_corners": True,
    }
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=None,
        weight_dict={},
        focal_alpha=0.25,
        losses=[],
        group_num=1,
        query_depth_cfg=query_config,
    )

    logits = torch.randn(
        batch_size, num_queries, num_classes, requires_grad=True
    )
    raw_probabilities = torch.rand(
        batch_size, num_queries, local_bins, requires_grad=True
    )
    probabilities = raw_probabilities / raw_probabilities.sum(
        dim=-1, keepdim=True
    )
    depth_centers = torch.linspace(8.0, 42.0, local_bins)
    bins = depth_centers.view(1, 1, -1).expand(
        batch_size, num_queries, -1
    )
    pred_depth = torch.stack(
        [
            torch.linspace(15.0, 32.0, num_queries),
            torch.zeros(num_queries),
        ],
        dim=-1,
    ).unsqueeze(0).repeat(batch_size, 1, 1)
    pred_depth.requires_grad_()
    pred_boxes = torch.tensor(
        [
            [
                [0.50, 0.50, 0.15, 0.15, 0.12, 0.12],
                [0.30, 0.45, 0.10, 0.12, 0.10, 0.14],
                [0.70, 0.55, 0.12, 0.13, 0.11, 0.12],
                [0.20, 0.30, 0.08, 0.08, 0.09, 0.09],
                [0.80, 0.40, 0.10, 0.10, 0.10, 0.10],
            ]
        ],
        dtype=torch.float32,
    ).repeat(batch_size, 1, 1)
    query_points = torch.rand(
        batch_size, num_queries, num_points, 2
    )
    point_weights = torch.full(
        (batch_size, num_queries, num_points, 1),
        1.0 / num_points,
    )
    gate = torch.full(
        (batch_size, num_queries, 1), 0.8, requires_grad=True
    )
    stereo_depth = pred_depth[..., :1]
    geometry_depth = stereo_depth.detach() + 1.5
    outputs = {
        "pred_logits": logits,
        "pred_boxes": pred_boxes,
        "pred_depth": pred_depth,
        "pred_depth_dist_probs": probabilities,
        "pred_depth_dist_bins": bins,
        "pred_depth_gate": gate,
        "pred_stereo_depth": stereo_depth,
        "pred_geometry_depth": geometry_depth,
        "pred_query_points": query_points,
        "pred_query_point_weights": point_weights,
    }
    targets = [
        {
            "labels": torch.tensor([0, 1]),
            "boxes_3d": pred_boxes[batch_index, :2].clone(),
            "depth": torch.tensor([[16.0], [24.0]]),
            "teacher_depth": torch.full((1, 24, 36), 20.0 + batch_index),
            "teacher_confidence": torch.full((1, 24, 36), 0.9),
        }
        for batch_index in range(batch_size)
    ]
    indices = [
        (torch.tensor([0, 1]), torch.tensor([0, 1]))
        for _ in range(batch_size)
    ]

    losses = {}
    losses.update(
        criterion.loss_labels(
            outputs, targets, indices, indices, num_boxes=4.0
        )
    )
    losses.update(
        criterion.loss_query_distribution(
            outputs, targets, indices, indices, num_boxes=4.0
        )
    )
    losses.update(
        criterion.loss_depth_gate(
            outputs, targets, indices, indices, num_boxes=4.0
        )
    )
    losses.update(
        criterion.loss_teacher_distill(
            outputs, targets, indices, indices, num_boxes=4.0
        )
    )
    optimized_losses = [
        value for key, value in losses.items() if key != "class_error"
    ]
    assert all(torch.isfinite(value).all() for value in optimized_losses)
    sum(optimized_losses).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert raw_probabilities.grad is not None
    assert torch.isfinite(raw_probabilities.grad).all()
    assert gate.grad is not None and torch.isfinite(gate.grad).all()
    print("质量对齐、分布、门控和离线教师损失测试通过")


if __name__ == "__main__":
    main()

