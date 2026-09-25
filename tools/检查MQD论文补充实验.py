#!/usr/bin/env python3
"""Static protocol audit for the MQD paper-evidence experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402


CONFIG_DIR = ROOT / "configs" / "MQD论文补充实验"
SHARED = (
    "S1_Hungarian匹配查询监督.yaml",
    "S2_全查询同类监督无尾部平衡.yaml",
    "S3_全查询同类监督加尾部平衡.yaml",
)
BOUNDARY = (
    "B1_普通Car残差.yaml",
    "B2_零初始化有界残差.yaml",
    "B3_边界加权决策场.yaml",
    "B4_完整边界决策场.yaml",
    "B4_完整边界决策场_种子445.yaml",
)


def main() -> None:
    shared = [load_config(CONFIG_DIR / name) for name in SHARED]
    for config in shared:
        quality = config["model"]["quality_ranking"]
        trainer = config["trainer"]
        dataset = config["dataset"]
        assert dataset["train_split"] == "train"
        assert dataset["test_split"] == "val"
        assert bool(quality["enabled"])
        assert bool(quality["freeze_detector"])
        assert not bool(quality["car_residual_enabled"])
        assert not bool(quality["pairwise_enabled"])
        assert int(trainer["max_epoch"]) == 5
        assert trainer["pretrain_model"].endswith(
            "/V08O_公平控制组/qecr_v08o_geometry_control/checkpoint_best.pth"
        )
        assert trainer["pretrain_allowed_missing_prefixes"] == ["quality_head."]
    assert shared[0]["model"]["quality_ranking"]["target_scope"] == "matched"
    assert shared[1]["model"]["quality_ranking"]["target_scope"] == "all"
    assert not bool(shared[1]["model"]["quality_ranking"]["tail_balance_enabled"])
    assert bool(shared[2]["model"]["quality_ranking"]["tail_balance_enabled"])

    boundary = [load_config(CONFIG_DIR / name) for name in BOUNDARY]
    for config in boundary:
        quality = config["model"]["quality_ranking"]
        trainer = config["trainer"]
        assert bool(quality["enabled"])
        assert bool(quality["freeze_detector"])
        assert bool(quality["car_residual_enabled"])
        assert bool(quality["car_residual_train_only"])
        assert int(trainer["max_epoch"]) == 3
        assert trainer["pretrain_model"].endswith(
            "/V09_V08O加点式三维质量排序/"
            "qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
        )
        assert trainer["pretrain_allowed_missing_prefixes"] == [
            "car_quality_head."
        ]
    b1, b2, b3, b4, b4_seed = boundary
    assert not bool(b1["model"]["quality_ranking"]["car_residual_bounded"])
    assert not bool(b1["model"]["quality_ranking"]["car_residual_zero_init"])
    assert bool(b2["model"]["quality_ranking"]["car_residual_bounded"])
    assert bool(b2["model"]["quality_ranking"]["car_residual_zero_init"])
    assert float(b2["model"]["quality_ranking"]["car_boundary_weight"]) == 0.0
    assert float(b3["model"]["quality_ranking"]["car_boundary_weight"]) == 2.0
    assert not bool(b3["model"]["quality_ranking"]["pairwise_enabled"])
    assert bool(b4["model"]["quality_ranking"]["pairwise_enabled"])
    assert int(b4_seed["random_seed"]) == 445

    smoke = load_config(CONFIG_DIR / "冒烟测试.yaml")
    assert int(smoke["trainer"]["max_train_batches"]) == 2
    assert int(smoke["trainer"]["max_epoch"]) == 1
    assert not bool(smoke["trainer"]["resume_if_exists"])

    # The shared head contains BatchNorm buffers.  The post-training audit
    # must not count running statistics as trainable parameters.
    audit_source = (ROOT / "tools" / "审计MQD补充实验权重.py").read_text(
        encoding="utf-8"
    )
    assert 'buffer_suffixes = ("running_mean", "running_var", "num_batches_tracked")' in audit_source
    assert "added_buffer_tensor_count" in audit_source

    model_source = (
        ROOT / "lib" / "models" / "monodetr" / "stereodetr.py"
    ).read_text(encoding="utf-8")
    assert 'if self.quality_target_scope == "matched":' in model_source
    assert "quality_indices, _ = self.matcher(" in model_source

    print("MQD论文补充实验协议检查通过")
    print("共享路径：匹配查询、全查询无尾部平衡、完整全查询监督")
    print("边界路径：普通残差、有界零初始化、边界加权、跨边界判别、第二种子")


if __name__ == "__main__":
    main()
