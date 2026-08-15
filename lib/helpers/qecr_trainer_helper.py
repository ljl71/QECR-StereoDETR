"""Trainer extension that preserves scalar stereo metadata per image."""

from lib.helpers.trainer_helper import Trainer


class QECRTrainer(Trainer):
    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets["mask_2d"]
        filtered_keys = {
            "labels",
            "boxes",
            "calibs",
            "depth",
            "size_3d",
            "src_size_3d",
            "heading_bin",
            "heading_res",
            "boxes_ry",
            "boxes_3d",
            "sample_points",
        }
        image_keys = {
            "disp",
            "disp_candidates",
            "disp_candidate_weights",
            "random_flip_flag",
            "random_crop_flag",
            "random_mix_flag",
            "random_switch_flag",
            "crop_scale",
            "stereo_fb",
            "stereo_direction",
        }
        for batch_index in range(batch_size):
            target_dict = {}
            for key, value in targets.items():
                if key in filtered_keys:
                    target_dict[key] = value[batch_index][mask[batch_index]]
                elif key in image_keys:
                    target_dict[key] = value[batch_index]
            targets_list.append(target_dict)
        return targets_list
