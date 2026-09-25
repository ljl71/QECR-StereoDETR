#!/usr/bin/env python3
"""Static audit for the full-trainval Car residual-quality experiment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402


CONFIG_DIR = ROOT / "configs" / "Car专项质量校准"


def main() -> None:
    train = load_config(CONFIG_DIR / "T3_trainval_Car残差质量.yaml")
    smoke = load_config(CONFIG_DIR / "Car残差质量冒烟.yaml")
    evaluate = load_config(CONFIG_DIR / "Car残差质量_KITTI测试.yaml")

    dataset = train["dataset"]
    trainer = train["trainer"]
    quality = train["model"]["quality_ranking"]
    assert dataset["train_split"] == "trainval"
    assert dataset["train_txt"].endswith("/training/ImageSets/trainval.txt")
    assert dataset["test_split"] == "test"
    assert dataset["eval_txt"].endswith("/testing/ImageSets/test.txt")
    assert int(dataset["batch_size"]) == 16
    assert int(trainer["max_epoch"]) == 3
    assert trainer["pretrain_model"].endswith(
        "/T2_QLQC/qlqc_trainval_global_batch16/checkpoint_final.pth"
    )
    assert trainer["pretrain_allowed_missing_prefixes"] == [
        "car_quality_head."
    ]
    assert trainer["pretrain_allow_unexpected"] is False
    assert bool(trainer["resume_if_exists"])
    assert bool(trainer["save_final_checkpoint"])

    assert bool(quality["enabled"])
    assert bool(quality["freeze_detector"])
    assert bool(quality["quality_only_when_frozen"])
    assert bool(quality["car_residual_enabled"])
    assert bool(quality["car_residual_train_only"])
    assert int(quality["car_class_index"]) == 0
    assert int(quality["car_residual_hidden_dim"]) == 32
    assert float(quality["car_residual_max_logit"]) == 0.5
    assert float(quality["car_iou_threshold"]) == 0.7
    assert bool(quality["pairwise_enabled"])
    assert not bool(train["tester"]["quality_guided_topk"])

    assert smoke["dataset"]["train_split"] == "train"
    assert int(smoke["trainer"]["max_epoch"]) == 1
    assert int(smoke["trainer"]["max_train_batches"]) == 2
    assert not bool(smoke["trainer"]["resume_if_exists"])

    assert evaluate["dataset"]["test_split"] == "test"
    assert int(evaluate["dataset"]["batch_size"]) == 12
    assert not bool(evaluate["trainer"]["amp_enabled"])
    assert not bool(evaluate["trainer"]["resume_if_exists"])

    print("Car专项质量校准配置检查通过：全量trainval、3轮、固定T2起点")
    print("训练范围检查通过：只允许car_quality_head.*参与优化")
    print("推理语义检查通过：分类Top-K不变，非Car继续使用原QLQC质量分数")


if __name__ == "__main__":
    main()
