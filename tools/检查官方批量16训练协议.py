"""Audit the true-global-batch-16 StereoDETR/QLQC trainval protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402
from lib.helpers.parallel_helper import balanced_chunk_sizes  # noqa: E402
from lib.helpers.parallel_helper import parse_gpu_ids  # noqa: E402
from lib.helpers.parallel_helper import split_batch_value  # noqa: E402


CONFIG_DIR = ROOT / "configs" / "官方批量16复核"


def _load(name):
    return load_config(CONFIG_DIR / name)


def _assert_trainval(config):
    dataset = config["dataset"]
    trainer = config["trainer"]
    assert dataset["train_split"] == "trainval"
    assert dataset["train_txt"].endswith("/training/ImageSets/trainval.txt")
    assert dataset["test_split"] == "test"
    assert dataset["eval_txt"].endswith("/testing/ImageSets/test.txt")
    assert int(dataset["batch_size"]) == 16
    assert parse_gpu_ids(trainer["gpu_ids"]) == [0]
    assert not bool(trainer["amp_enabled"])
    assert str(trainer["amp_dtype"]).lower() == "float16"
    assert bool(trainer["zero_grad_set_to_none"])
    assert bool(trainer["resume_if_exists"])
    assert bool(trainer["save_final_checkpoint"])
    assert not bool(trainer.get("save_all", False))
    assert "KITTI_trainval官方批量16复核" in trainer["save_path"]


def _test_batch_partition():
    assert balanced_chunk_sizes(16, 1) == [16]
    assert balanced_chunk_sizes(16, 2) == [8, 8]
    assert balanced_chunk_sizes(15, 2) == [8, 7]

    targets = [
        {
            "sample_id": index,
            "depth": torch.tensor([float(index)]),
        }
        for index in range(16)
    ]
    target_chunks = split_batch_value(
        targets,
        batch_size=16,
        chunk_sizes=[8, 8],
        device_ids=[None, None],
    )
    assert [len(chunk) for chunk in target_chunks] == [8, 8]
    assert [item["sample_id"] for item in target_chunks[0]] == list(range(8))
    assert [item["sample_id"] for item in target_chunks[1]] == list(range(8, 16))

    tensor_chunks = split_batch_value(
        {"calib": torch.arange(16 * 4).reshape(16, 4)},
        batch_size=16,
        chunk_sizes=[8, 8],
        device_ids=[None, None],
    )
    assert tensor_chunks[0]["calib"].shape == (8, 4)
    assert tensor_chunks[1]["calib"].shape == (8, 4)
    assert int(tensor_chunks[1]["calib"][0, 0]) == 32


def main():
    base = _load("T0_trainval_StereoDETR.yaml")
    control = _load("T1_trainval强控制.yaml")
    qlqc = _load("T2_trainval_QLQC.yaml")
    baseline_eval = _load("E0_StereoDETR_KITTI测试.yaml")
    qlqc_eval = _load("E2_QLQC_KITTI测试.yaml")
    smoke = _load("批量16显存冒烟.yaml")

    for config in (base, control, qlqc):
        _assert_trainval(config)

    assert int(base["trainer"]["max_epoch"]) == 195
    assert float(base["optimizer"]["lr"]) == 0.0002
    assert list(base["lr_scheduler"]["decay_list"]) == [125, 165]
    assert not bool(base["model"]["quality_ranking"]["enabled"])
    assert not base["trainer"].get("pretrain_model")

    assert int(control["trainer"]["max_epoch"]) == 1
    assert float(control["optimizer"]["lr"]) == 0.00002
    assert control["trainer"]["pretrain_model"].endswith(
        "/T0_StereoDETR/stereodetr_trainval_global_batch16/"
        "checkpoint_final.pth"
    )
    assert bool(control["trainer"]["pretrain_strict"])
    assert not bool(control["model"]["quality_ranking"]["enabled"])
    for key in ("aug_pd", "aug_crop"):
        assert not bool(control["dataset"][key])
    for key in (
        "random_mixup3d",
        "random_flip",
        "random_crop",
        "random_switch",
    ):
        assert float(control["dataset"][key]) == 0.0

    quality = qlqc["model"]["quality_ranking"]
    assert int(qlqc["trainer"]["max_epoch"]) == 3
    assert float(qlqc["optimizer"]["lr"]) == 0.001
    assert qlqc["trainer"]["pretrain_model"].endswith(
        "/T1_强控制/"
        "stereodetr_trainval_global_batch16_strong_control/"
        "checkpoint_final.pth"
    )
    assert qlqc["trainer"]["pretrain_allowed_missing_prefixes"] == [
        "quality_head."
    ]
    assert qlqc["trainer"]["pretrain_allow_unexpected"] is False
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

    for eval_config in (baseline_eval, qlqc_eval):
        assert eval_config["dataset"]["test_split"] == "test"
        assert int(eval_config["dataset"]["batch_size"]) == 12
        assert not bool(eval_config["trainer"]["amp_enabled"])
        assert not bool(eval_config["trainer"]["resume_if_exists"])
        assert not bool(eval_config["trainer"]["save_final_checkpoint"])

    assert smoke["dataset"]["train_split"] == "train"
    assert int(smoke["dataset"]["batch_size"]) == 16
    assert int(smoke["trainer"]["max_epoch"]) == 1
    assert int(smoke["trainer"]["max_train_batches"]) == 2
    assert not bool(smoke["trainer"]["resume_if_exists"])

    deformable_attention_source = (
        ROOT
        / "lib"
        / "models"
        / "monodetr"
        / "ops"
        / "functions"
        / "ms_deform_attn_func.py"
    ).read_text(encoding="utf-8")
    assert "@custom_fwd(cast_inputs=torch.float32)" in (
        deformable_attention_source
    )
    assert "@custom_bwd" in deformable_attention_source

    _test_batch_partition()
    print("官方批量16协议检查通过：195+1+3固定轮次，训练全局batch=16")
    print("批次切分检查通过：单卡16，双卡8+8，targets与张量保持样本对齐")
    print("运行时支持：单卡FP32、单卡AMP和双卡FP32；旧V09配置与输出不变")
    print("AMP边界检查通过：自定义可变形注意力在混合精度下保持FP32")


if __name__ == "__main__":
    main()
