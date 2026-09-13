"""Regression test for half-output/FP32 Hungarian matching under AMP."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MATCHER_PATH = (
    ROOT / "lib" / "models" / "monodetr" / "matcherstable_fg_only.py"
)


def _load_matcher_class():
    # Loading the source directly keeps this small test independent of the
    # compiled deformable-attention extension.
    spec = importlib.util.spec_from_file_location(
        "qecr_amp_matcher_test_module",
        MATCHER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StableHungarianMatcher


def main():
    matcher_class = _load_matcher_class()
    matcher = matcher_class(
        cost_class=2,
        cost_3dcenter=0,
        cost_depth=0,
        cost_bbox=5,
        cost_giou=2,
        cec_beta=0.5,
        iou_thresh=0.02,
    )
    target_boxes = torch.tensor(
        [
            [0.50, 0.50, 0.10, 0.10, 0.10, 0.10],
            [0.40, 0.40, 0.10, 0.10, 0.10, 0.10],
        ],
        dtype=torch.float32,
    )
    targets = [
        {
            "labels": torch.tensor([0, 2]),
            "boxes": target_boxes.clone(),
            "boxes_3d": target_boxes.clone(),
            "depth": torch.tensor([[10.0], [20.0]]),
        },
        {
            "labels": torch.tensor([1]),
            "boxes": target_boxes[:1].clone(),
            "boxes_3d": target_boxes[:1].clone(),
            "depth": torch.tensor([[15.0]]),
        },
    ]
    outputs = {
        "pred_logits": torch.randn(2, 11, 4, dtype=torch.float16),
        "pred_boxes": (
            torch.rand(2, 11, 6, dtype=torch.float16) * 0.2 + 0.4
        ),
        "pred_depth": (
            torch.rand(2, 11, 1, dtype=torch.float16) * 20.0 + 1.0
        ),
    }
    indices, filtered = matcher(outputs, targets, group_num=11)
    assert len(indices) == len(filtered) == 2
    print("AMP匹配器测试通过：Half模型输出使用FP32代价完成Hungarian匹配")


if __name__ == "__main__":
    main()
