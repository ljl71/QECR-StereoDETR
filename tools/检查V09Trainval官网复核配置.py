"""Audit the fixed, no-test-tuning V09 trainval submission protocol."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402


CONFIG_DIR = ROOT / "configs" / "V09_trainval官网复核"


def _load(name):
    return load_config(CONFIG_DIR / name)


def _assert_common_trainval(config):
    dataset = config["dataset"]
    trainer = config["trainer"]
    assert dataset["train_split"] == "trainval"
    assert dataset["train_txt"].endswith("/training/ImageSets/trainval.txt")
    assert dataset["test_split"] == "test"
    assert dataset["eval_txt"].endswith("/testing/ImageSets/test.txt")
    assert dataset["root_dir"].endswith("/object/training")
    assert dataset["root_dir_eval"].endswith("/object/testing")
    assert int(dataset["batch_size"]) == 12
    assert bool(trainer["resume_if_exists"])
    assert bool(trainer["save_final_checkpoint"])
    assert not bool(trainer.get("save_all", False))


def main():
    base = _load("V09T0_trainval基础模型.yaml")
    control = _load("V09T1_trainval强控制.yaml")
    qlqc = _load("V09T2_trainval_QLQC.yaml")
    qlqc_eval = _load("V09E2_QLQC官网测试.yaml")

    for config in (base, control, qlqc):
        _assert_common_trainval(config)

    assert base["model_name"] == "qecr_v09t0_trainval_stereodetr"
    assert int(base["trainer"]["max_epoch"]) == 195
    assert not bool(base["model"]["quality_ranking"]["enabled"])
    assert not base["trainer"].get("pretrain_model")

    control_dataset = control["dataset"]
    control_trainer = control["trainer"]
    assert control["model_name"] == "qecr_v09t1_trainval_strong_control"
    assert int(control_trainer["max_epoch"]) == 1
    assert float(control["optimizer"]["lr"]) == 0.00002
    assert control_trainer["pretrain_model"].endswith(
        "/V09T0_trainval基础模型/"
        "qecr_v09t0_trainval_stereodetr/checkpoint_final.pth"
    )
    assert bool(control_trainer["pretrain_strict"])
    assert not bool(control["model"]["quality_ranking"]["enabled"])
    for key in ("aug_pd", "aug_crop"):
        assert not bool(control_dataset[key])
    for key in (
        "random_mixup3d",
        "random_flip",
        "random_crop",
        "random_switch",
    ):
        assert float(control_dataset[key]) == 0.0

    quality = qlqc["model"]["quality_ranking"]
    qlqc_trainer = qlqc["trainer"]
    assert qlqc["model_name"] == "qecr_v09t2_trainval_qlqc"
    assert int(qlqc_trainer["max_epoch"]) == 3
    assert float(qlqc["optimizer"]["lr"]) == 0.001
    assert qlqc_trainer["pretrain_model"].endswith(
        "/V09T1_trainval强控制/"
        "qecr_v09t1_trainval_strong_control/checkpoint_final.pth"
    )
    assert qlqc_trainer["pretrain_strict"] is False
    assert qlqc_trainer["pretrain_allowed_missing_prefixes"] == [
        "quality_head."
    ]
    assert qlqc_trainer["pretrain_allow_unexpected"] is False
    assert bool(quality["enabled"])
    assert bool(quality["loss_enabled"])
    assert bool(quality["freeze_detector"])
    assert bool(quality["quality_only_when_frozen"])
    assert not bool(quality["pairwise_enabled"])
    assert float(quality["score_power"]) == 1.5
    assert float(quality["point_loss_coef"]) == 10.0
    assert float(quality["negative_threshold"]) == 0.1
    assert float(quality["negative_weight"]) == 0.1
    assert not bool(qlqc["tester"].get("quality_guided_topk", False))

    assert qlqc_eval["model_name"] == (
        "qecr_v09e2_trainval_qlqc_kitti_test"
    )
    assert qlqc_eval["dataset"]["test_split"] == "test"
    assert qlqc_eval["trainer"]["save_path"].endswith(
        "/官网测试/QLQC最终模型/"
    )
    assert not bool(qlqc_eval["trainer"]["resume_if_exists"])
    assert not bool(qlqc_eval["trainer"]["save_final_checkpoint"])

    print("V09 trainval官网复核配置检查通过：195轮基础模型、1轮强控制、3轮QLQC均为固定轮次协议")
    print("最终官网测试配置只读取固定QLQC权重，不使用KITTI test标签选优或调参")


if __name__ == "__main__":
    main()
