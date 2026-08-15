"""Validate P2/P3 baseline conversion and augmentation-scale convention."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.datasets.kitti.qecr_kitti_dataset import QECRKITTIDataset  # noqa: E402


def main():
    focal = 700.0
    baseline = 0.54
    p2 = np.array(
        [
            [focal, 0.0, 600.0, 0.0],
            [0.0, focal, 180.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    p3 = p2.copy()
    p3[0, 3] = -focal * baseline

    dataset = QECRKITTIDataset.__new__(QECRKITTIDataset)
    dataset.depth_scale = "normal"
    dataset.resolution_ori = np.array([1280, 388])
    dataset.get_calib = lambda _index: SimpleNamespace(P2=p2, P3=p3)

    original_width = 1242.0
    network_fb = float(dataset._stereo_fb("000000", original_width))
    expected_fb = focal * baseline * 1280.0 / original_width
    assert abs(network_fb - expected_fb) < 1.0e-4

    original_depth = 20.0
    crop_scale = 1.2
    augmented_depth = original_depth * crop_scale
    disparity_from_qecr = network_fb / augmented_depth
    disparity_from_image_transform = (
        focal
        * baseline
        / original_depth
        * 1280.0
        / (original_width * crop_scale)
    )
    assert abs(
        disparity_from_qecr - disparity_from_image_transform
    ) < 1.0e-5

    assert dataset._direction_from_target({}) == -1.0
    assert dataset._direction_from_target(
        {"random_flip_flag": True, "random_switch_flag": False}
    ) == 1.0
    assert dataset._direction_from_target(
        {"random_flip_flag": False, "random_switch_flag": True}
    ) == 1.0
    assert dataset._direction_from_target(
        {"random_flip_flag": True, "random_switch_flag": True}
    ) == -1.0
    print("P2/P3 双目标定、裁剪尺度与方向测试通过")


if __name__ == "__main__":
    main()

