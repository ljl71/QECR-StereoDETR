#!/usr/bin/env python3
"""CPU tests for the V06 quality head, losses, targets and decode score."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.decode_helper import extract_dets_from_outputs  # noqa: E402
from lib.models.monodetr.quality_ranking import (  # noqa: E402
    CarResidualQualityHead,
    Query3DQualityHead,
    build_3d_iou_quality_targets,
    car_boundary_pairwise_quality_loss,
    car_focused_quality_loss,
    pairwise_quality_loss,
    pointwise_quality_loss,
    quality_probability_from_logits,
    quality_supervision_mask,
)


def exact_match_overlap(predicted, target):
    result = np.zeros((len(predicted), len(target)), dtype=np.float64)
    for pred_index, pred_box in enumerate(predicted):
        for target_index, target_box in enumerate(target):
            if np.allclose(pred_box, target_box, atol=1.0e-6):
                result[pred_index, target_index] = 1.0
    return result


def main():
    torch.manual_seed(7)
    head = Query3DQualityHead(256)
    assert sum(parameter.numel() for parameter in head.parameters()) == 66561
    features = torch.randn(2, 5, 256, requires_grad=True)
    logits = head(features)
    targets = torch.linspace(0.0, 0.9, 10).reshape(2, 5)
    labels = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]])
    loss = pointwise_quality_loss(logits, targets)
    loss = loss + pairwise_quality_loss(logits, targets, labels)
    loss.backward()
    assert torch.isfinite(features.grad).all()

    car_head = CarResidualQualityHead(
        hidden_dim=256,
        bottleneck_dim=32,
        max_logit_residual=0.5,
    )
    assert sum(parameter.numel() for parameter in car_head.parameters()) == 8257
    residual = car_head(features.detach())
    assert torch.equal(residual, torch.zeros_like(residual))
    base_logits = torch.randn(2, 5, 1)
    car_mask = (labels == 0).unsqueeze(-1)
    calibrated = base_logits + torch.where(
        car_mask, residual, torch.zeros_like(residual)
    )
    assert torch.equal(calibrated[~car_mask], base_logits[~car_mask])

    ordinary_car_head = CarResidualQualityHead(
        hidden_dim=256,
        bottleneck_dim=32,
        max_logit_residual=0.5,
        bounded=False,
        zero_init=False,
    )
    ordinary_residual = ordinary_car_head(features.detach())
    assert torch.count_nonzero(ordinary_residual) > 0

    matched_indices = [
        (torch.tensor([1, 3]), torch.tensor([0, 1])),
        (torch.tensor([0]), torch.tensor([0])),
    ]
    matched_mask = quality_supervision_mask(
        targets,
        matched_indices=matched_indices,
        target_scope="matched",
    )
    assert torch.equal(
        matched_mask,
        torch.tensor(
            [[False, True, False, True, False],
             [True, False, False, False, False]]
        ),
    )
    all_mask = quality_supervision_mask(targets, target_scope="all")
    assert all_mask.all()
    masked_logits = torch.zeros(2, 5, 1, requires_grad=True)
    masked_loss = pointwise_quality_loss(
        masked_logits,
        targets,
        valid_mask=matched_mask,
    )
    masked_loss.backward()
    assert torch.count_nonzero(masked_logits.grad[~matched_mask]) == 0

    boundary_logits = torch.tensor(
        [[[0.0], [0.1], [-0.2], [0.4]], [[-0.1], [0.2], [0.3], [-0.4]]],
        requires_grad=True,
    )
    boundary_targets = torch.tensor(
        [[0.82, 0.58, 0.75, 0.20], [0.90, 0.40, 0.72, 0.05]]
    )
    boundary_labels = torch.tensor([[0, 0, 1, 1], [0, 0, 2, 2]])
    car_loss = car_focused_quality_loss(
        boundary_logits,
        boundary_targets,
        boundary_labels,
    )
    car_loss = car_loss + car_boundary_pairwise_quality_loss(
        boundary_logits,
        boundary_targets,
        boundary_labels,
    )
    car_loss.backward()
    assert torch.isfinite(boundary_logits.grad).all()
    assert torch.count_nonzero(boundary_logits.grad[boundary_labels != 0]) == 0

    calibration_logits = torch.tensor([-2.0, 0.0, 2.0])
    base_quality = calibration_logits.sigmoid()
    assert torch.allclose(
        quality_probability_from_logits(calibration_logits, 0.0),
        torch.ones_like(base_quality),
    )
    assert torch.allclose(
        quality_probability_from_logits(calibration_logits, 0.5),
        base_quality.sqrt(),
    )
    assert torch.allclose(
        quality_probability_from_logits(calibration_logits, 1.0),
        base_quality,
    )
    try:
        quality_probability_from_logits(calibration_logits, -0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative quality score power must be rejected")

    outputs = {
        "pred_logits": torch.tensor([[[4.0, -4.0], [3.0, -3.0]]]),
        "pred_boxes": torch.tensor(
            [[[0.5, 0.5, 0.1, 0.1, 0.1, 0.1], [0.6, 0.5, 0.1, 0.1, 0.1, 0.1]]]
        ),
        "pred_sample_points": torch.tensor([[[0.5, 0.5], [0.6, 0.5]]]),
        "pred_depth": torch.tensor([[[10.0, 0.0], [12.0, 0.0]]]),
        "pred_3d_dim": torch.tensor([[[1.5, 1.6, 3.8], [1.5, 1.6, 3.8]]]),
        "pred_angle": torch.zeros(1, 2, 24),
        "pred_quality": torch.full((1, 2, 1), 0.5),
        "quality_calibs": torch.tensor(
            [[[700.0, 0.0, 620.0, 0.0], [0.0, 700.0, 180.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
        ),
        "quality_img_sizes": torch.tensor([[1280.0, 288.0]]),
        "quality_img_sizes_ori": torch.tensor([[1242.0, 375.0]]),
        "quality_img_sizes_upper": torch.tensor([100.0]),
    }
    training_targets = [{
        "labels": torch.tensor([0]),
        "boxes_3d": torch.tensor([[0.5, 0.5, 0.1, 0.1, 0.1, 0.1]]),
        "depth": torch.tensor([[10.0]]),
        "size_3d": torch.tensor([[1.5, 1.6, 3.8]]),
        "heading_bin": torch.tensor([[0]]),
        "heading_res": torch.tensor([[0.0]]),
        "random_flip_flag": torch.tensor(False),
        "random_crop_flag": torch.tensor(False),
        "random_mix_flag": torch.tensor(False),
        "random_switch_flag": torch.tensor(False),
        "crop_scale": torch.tensor(1.0),
    }]
    quality_targets = build_3d_iou_quality_targets(
        outputs, training_targets, overlap_fn=exact_match_overlap
    )
    assert torch.allclose(quality_targets, torch.tensor([[1.0, 0.0]]))

    detections = extract_dets_from_outputs(outputs, K=2, topk=2)
    assert torch.allclose(detections[..., -1], torch.full((1, 2), 0.5))

    routed_outputs = dict(outputs)
    routed_outputs["pred_quality_base"] = torch.full((1, 2, 1), 0.4)
    routed_outputs["pred_quality_car"] = torch.full((1, 2, 1), 0.9)
    routed_outputs["quality_car_class_index"] = 0
    routed_detections = extract_dets_from_outputs(
        routed_outputs,
        K=4,
        topk=4,
    )
    routed_labels = routed_detections[..., 0].long()
    routed_factors = routed_detections[..., -1]
    assert torch.allclose(
        routed_factors[routed_labels == 0],
        torch.full_like(routed_factors[routed_labels == 0], 0.9),
    )
    assert torch.allclose(
        routed_factors[routed_labels != 0],
        torch.full_like(routed_factors[routed_labels != 0], 0.4),
    )

    retrieval_outputs = dict(outputs)
    retrieval_outputs["pred_quality"] = torch.tensor(
        [[[0.01], [1.0]]]
    )
    class_top1 = extract_dets_from_outputs(
        retrieval_outputs,
        K=1,
        topk=1,
        quality_guided_topk=False,
    )
    guided_top1 = extract_dets_from_outputs(
        retrieval_outputs,
        K=1,
        topk=1,
        quality_guided_topk=True,
    )
    # Class-only retrieval selects query 0 (depth 10), whereas the aligned
    # 3D-quality score retrieves query 1 (depth 12).  The stored class score
    # remains the original probability, so quality is not multiplied twice.
    assert torch.allclose(class_top1[..., 6], torch.tensor([[10.0]]))
    assert torch.allclose(guided_top1[..., 6], torch.tensor([[12.0]]))
    assert torch.allclose(
        guided_top1[..., 1],
        torch.tensor([[3.0]]).sigmoid(),
    )
    assert torch.allclose(guided_top1[..., -1], torch.ones(1, 1))
    try:
        without_quality = dict(retrieval_outputs)
        without_quality.pop("pred_quality")
        extract_dets_from_outputs(
            without_quality,
            K=1,
            topk=1,
            quality_guided_topk=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("guided Top-K must require a learned quality score")

    print(
        "V06/V07质量排序与Car类别残差路由测试通过"
    )


if __name__ == "__main__":
    main()
