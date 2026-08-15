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
    Query3DQualityHead,
    build_3d_iou_quality_targets,
    pairwise_quality_loss,
    pointwise_quality_loss,
    quality_probability_from_logits,
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
        "V06质量头/监督与V07质量引导Top-K检索测试通过"
    )


if __name__ == "__main__":
    main()
