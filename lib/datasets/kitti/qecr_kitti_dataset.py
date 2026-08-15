"""KITTI dataset extension that exposes physically valid stereo metadata."""

from __future__ import annotations

import numpy as np

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset


class QECRKITTIDataset(KITTI_Dataset):
    """Add network-scale ``f_x * baseline`` and epipolar direction per sample."""

    def _stereo_fb(self, index: str, original_width: float) -> np.float32:
        if self.depth_scale != "normal":
            raise ValueError(
                "QECR currently requires dataset.depth_scale='normal' so "
                "network-scale disparity and augmented depth remain consistent"
            )
        calibration = self.get_calib(index)
        p2 = calibration.P2.astype(np.float64)
        p3 = calibration.P3.astype(np.float64)
        if abs(p2[0, 0]) < 1.0e-9 or abs(p3[0, 0]) < 1.0e-9:
            raise ValueError("invalid KITTI stereo focal length")

        camera_center_2 = -p2[0, 3] / p2[0, 0]
        camera_center_3 = -p3[0, 3] / p3[0, 0]
        baseline = abs(camera_center_3 - camera_center_2)
        if baseline <= 0.0:
            raise ValueError("KITTI P2/P3 do not define a positive stereo baseline")

        # Images are resized to resolution_ori.  With depth_scale='normal',
        # random crop scale is absorbed into the training depth target, so it
        # must not be included a second time in this fB value.
        width_scale_without_random_crop = (
            float(self.resolution_ori[0]) / max(float(original_width), 1.0)
        )
        network_fb = p2[0, 0] * baseline * width_scale_without_random_crop
        return np.float32(network_fb)

    @staticmethod
    def _direction_from_target(targets) -> np.float32:
        if not isinstance(targets, dict):
            return np.float32(-1.0)
        flipped = bool(targets.get("random_flip_flag", False))
        switched = bool(targets.get("random_switch_flag", False))
        # Normal left-to-right correspondence is x_right = x_left - disparity.
        # A simultaneous horizontal flip or one left/right switch reverses it.
        reversed_once = flipped != switched
        return np.float32(1.0 if reversed_once else -1.0)

    def __getitem__(self, item):
        inputs, calibration, targets, info = super().__getitem__(item)
        index = self.idx_list[item]
        original_width = float(info["img_size_original"][0])
        stereo_fb = self._stereo_fb(index, original_width)
        stereo_direction = self._direction_from_target(targets)

        if isinstance(targets, dict):
            targets["stereo_fb"] = stereo_fb
            targets["stereo_direction"] = stereo_direction
        else:
            # Official KITTI test split returns no labels.  The base model does
            # not consume targets in eval mode, while QECR still needs fB.
            targets = {
                "stereo_fb": stereo_fb,
                "stereo_direction": stereo_direction,
            }

        info["stereo_fb"] = stereo_fb
        info["stereo_direction"] = stereo_direction
        return inputs, calibration, targets, info

