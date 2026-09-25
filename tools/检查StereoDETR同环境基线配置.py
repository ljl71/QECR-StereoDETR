"""Audit the same-environment StereoDETR trainval/test baseline protocol."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402


CONFIG_DIR = ROOT / "configs" / "V09_trainval官网复核"
TRAIN_CONFIG = CONFIG_DIR / "V09T0_trainval基础模型.yaml"
EVAL_CONFIG = CONFIG_DIR / "V09E0_StereoDETR基线官网测试.yaml"


def _assert_disabled(model, key):
    config = model.get(key, {})
    assert isinstance(config, dict), f"model.{key}必须是映射"
    assert not bool(config.get("enabled", False)), f"model.{key}必须关闭"


def main():
    train = load_config(TRAIN_CONFIG)
    evaluate = load_config(EVAL_CONFIG)

    dataset = train["dataset"]
    trainer = train["trainer"]
    assert train["model_name"] == "qecr_v09t0_trainval_stereodetr"
    assert dataset["train_split"] == "trainval"
    assert dataset["train_txt"].endswith("/training/ImageSets/trainval.txt")
    assert dataset["test_split"] == "test"
    assert dataset["eval_txt"].endswith("/testing/ImageSets/test.txt")
    assert dataset["root_dir"].endswith("/object/training")
    assert dataset["root_dir_eval"].endswith("/object/testing")
    assert int(dataset["batch_size"]) == 12
    assert int(trainer["max_epoch"]) == 195
    assert bool(trainer["resume_if_exists"])
    assert bool(trainer["save_final_checkpoint"])
    assert not bool(trainer.get("save_all", False))
    assert not trainer.get("pretrain_model")

    model = train["model"]
    for key in (
        "depth_readout",
        "dynamic_depth_upsampling",
        "groupwise_correlation",
        "correlation_smoothing",
        "cost_preaggregation",
        "geometry_depth_residual",
        "axial_depth_iou",
        "query_depth",
        "query_epipolar",
        "quality_ranking",
        "geometry_alignment",
    ):
        _assert_disabled(model, key)
    assert not bool(train["tester"].get("quality_guided_topk", False))

    eval_trainer = evaluate["trainer"]
    assert evaluate["model_name"] == (
        "qecr_v09e0_stereodetr_reimpl_kitti_test"
    )
    assert evaluate["dataset"]["test_split"] == "test"
    assert evaluate["dataset"]["eval_txt"].endswith(
        "/testing/ImageSets/test.txt"
    )
    assert evaluate["model"]["quality_ranking"]["enabled"] is False
    assert not bool(evaluate["tester"].get("quality_guided_topk", False))
    assert eval_trainer["save_path"].endswith(
        "/官网测试/StereoDETR同环境基线/"
    )
    assert not bool(eval_trainer["resume_if_exists"])
    assert not bool(eval_trainer["save_final_checkpoint"])

    print("StereoDETR同环境基线配置检查通过")
    print("协议：trainval 7481张、batch 12、195轮、全部QECR/QLQC实验模块关闭")
    print("官网推理：固定T0 checkpoint_final，不训练、不选优、不读取test标签")


if __name__ == "__main__":
    main()
