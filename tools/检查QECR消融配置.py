"""Verify the non-distillation ablation and low-cost diagnostic matrix."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402


EXPECTED = {
    "V00_官方基线": (False, False, False, 1, False, False, False),
    "V01_查询级局部分布": (True, False, False, 1, False, False, False),
    "V02_跨层分布细化": (True, True, False, 1, False, False, False),
    "V03_基线安全查询深度": (True, False, True, 1, False, False, False),
    "V04_基线安全跨层细化": (True, True, True, 1, False, False, False),
    "V15_单向局部极线校正": (True, True, False, 1, True, False, False),
    "V16_双向极线一致性": (True, True, False, 1, True, True, False),
    "V17_双向一致性加多点采样": (True, True, False, 5, True, True, False),
    "V18_无蒸馏完整模型": (True, True, False, 5, True, True, True),
}

EXPECTED_DISPARITY = {
    "V05A_短程续训对照": ("StereoBM", "scalar", False),
    "V05B_SGBM单值修正": ("SGBM", "scalar", False),
    "V05C_SGBM多候选监督": ("SGBM", "multi_candidate", False),
    "V05D_多候选加LRC软权重": ("SGBM", "multi_candidate", True),
}

EXPECTED_QUALITY = {
    "V06B_点式三维质量排序": (False, False),
    "V06C_点式加成对排序": (True, False),
    "V07_质量引导候选检索": (False, True),
}

EXPECTED_GEOMETRY = {
    "V08O_几何对齐短程对照": (False, False, False),
    "V08A_分阶段三维角点对齐": (True, True, False),
    "V08B_分阶段左目投影对齐": (True, False, True),
    "V08C_空间投影协同对齐": (True, True, True),
}

EXPECTED_GEOMETRY_OUTPUTS = {
    "V08O_几何对齐短程对照": "outputs/第二创新点_V08空间投影对齐/V08O_公平控制组/",
    "V08A_分阶段三维角点对齐": "outputs/第二创新点_V08空间投影对齐/V08A_三维角点对齐/",
    "V08B_分阶段左目投影对齐": "outputs/第二创新点_V08空间投影对齐/V08B_左目投影对齐/",
    "V08C_空间投影协同对齐": "outputs/第二创新点_V08空间投影对齐/V08C_空间投影协同对齐/",
}

EXPECTED_FINAL_COMBINATIONS = {
    "V09_V08O加点式三维质量排序": {
        "pretrain_model": (
            "/root/autodl-tmp/QECR-StereoDETR/outputs/"
            "第二创新点_V08空间投影对齐/V08O_公平控制组/"
            "qecr_v08o_geometry_control/checkpoint_best.pth"
        ),
        "save_path": (
            "outputs/最终组合_V09/V09_V08O加点式三维质量排序/"
        ),
    },
}

EXPECTED_DEPTH_READOUT_DIAGNOSTICS = {
    "V10O_全分布深度读出控制": {
        "enabled": False,
        "top_k": 2,
        "model_name": "qecr_v10o_full_distribution_control",
        "save_path": "outputs/第二创新点_V10前景TopK深度读出诊断/V10O_全分布控制/",
    },
    "V10A_前景TopK2深度读出": {
        "enabled": True,
        "top_k": 2,
        "model_name": "qecr_v10a_foreground_topk2",
        "save_path": "outputs/第二创新点_V10前景TopK深度读出诊断/V10A_前景TopK2/",
    },
    "V10B_前景TopK4深度读出": {
        "enabled": True,
        "top_k": 4,
        "model_name": "qecr_v10b_foreground_topk4",
        "save_path": "outputs/第二创新点_V10前景TopK深度读出诊断/V10B_前景TopK4/",
    },
}

EXPECTED_DYNAMIC_DEPTH_UPSAMPLING = {
    "V11O_双线性深度上采样控制": {
        "enabled": False,
        "stages": [],
        "strict": True,
        "missing": [],
        "model_name": "qecr_v11o_bilinear_depth_upsampling_control",
        "save_path": "outputs/第二创新点_V11动态深度上采样/V11O_双线性训练控制/",
    },
    "V11A_第一级动态深度上采样": {
        "enabled": True,
        "stages": [1],
        "strict": False,
        "missing": [
            "depth_predictor.depth_classifier.0.offset_predictor."
        ],
        "model_name": "qecr_v11a_stage1_dynamic_depth_upsampling",
        "save_path": "outputs/第二创新点_V11动态深度上采样/V11A_第一级动态上采样/",
    },
    "V11B_两级动态深度上采样": {
        "enabled": True,
        "stages": [1, 2],
        "strict": False,
        "missing": [
            "depth_predictor.depth_classifier.0.offset_predictor.",
            "depth_predictor.depth_classifier.4.offset_predictor.",
        ],
        "model_name": "qecr_v11b_two_stage_dynamic_depth_upsampling",
        "save_path": "outputs/第二创新点_V11动态深度上采样/V11B_两级动态上采样/",
    },
}

EXPECTED_GROUPWISE_CORRELATION = {
    "V12O_原始相关公平控制": {
        "seed": 444,
        "enabled": False,
        "strict": True,
        "missing": [],
        "prefixes": ["depth_predictor.cost_agg."],
        "model_name": "qecr_v12o_original_correlation_control",
        "save_path": "outputs/第二创新点_V12分组相关门控/V12O_原始相关公平控制/",
    },
    "V12A_s4十六组轻量门控": {
        "seed": 444,
        "enabled": True,
        "strict": False,
        "missing": ["depth_predictor.groupwise_correlation_s4."],
        "prefixes": [
            "depth_predictor.cost_agg.",
            "depth_predictor.groupwise_correlation_s4.",
        ],
        "model_name": "qecr_v12a_s4_group16_light_gate",
        "save_path": "outputs/第二创新点_V12分组相关门控/V12A_s4十六组轻量门控/",
    },
    "V12O_种子445原始相关控制": {
        "seed": 445,
        "enabled": False,
        "strict": True,
        "missing": [],
        "prefixes": ["depth_predictor.cost_agg."],
        "model_name": "qecr_v12o_seed445_original_correlation_control",
        "save_path": "outputs/第二创新点_V12分组相关门控/第二随机种子_445/V12O_原始相关控制/",
    },
    "V12A_种子445s4十六组轻量门控": {
        "seed": 445,
        "enabled": True,
        "strict": False,
        "missing": ["depth_predictor.groupwise_correlation_s4."],
        "prefixes": [
            "depth_predictor.cost_agg.",
            "depth_predictor.groupwise_correlation_s4.",
        ],
        "model_name": "qecr_v12a_seed445_s4_group16_light_gate",
        "save_path": "outputs/第二创新点_V12分组相关门控/第二随机种子_445/V12A_s4十六组轻量门控/",
    },
}

EXPECTED_FIXED_SMOOTHING_DIAGNOSTICS = {
    "V14O_零训练原始相关复评": {
        "enabled": False,
        "model_name": "qecr_v14o_zero_train_original_correlation",
        "save_path": "outputs/固定相关平滑_V14/零训练复评/V14O_原始相关/",
    },
    "V14S_零训练s4固定平滑": {
        "enabled": True,
        "model_name": "qecr_v14s_zero_train_s4_fixed_smoothing",
        "save_path": "outputs/固定相关平滑_V14/零训练复评/V14S_s4固定平滑/",
    },
}

EXPECTED_GEOMETRY_DEPTH_RESIDUAL = {
    "V20O_V09零训练深度控制": {
        "enabled": False,
        "use_geometry": False,
        "train_only": False,
        "strict": True,
        "missing": [],
        "model_name": "qecr_v20o_v09_zero_train_depth_control",
        "save_path": "outputs/第三阶段_V20立体几何深度残差/V20O_V09零训练深度控制/",
    },
    "V20A_查询级深度残差": {
        "enabled": True,
        "use_geometry": False,
        "train_only": True,
        "strict": False,
        "missing": ["depth_residual_head."],
        "model_name": "qecr_v20a_query_only_depth_residual",
        "save_path": "outputs/第三阶段_V20立体几何深度残差/V20A_查询级深度残差/",
    },
    "V20B_立体几何深度残差": {
        "enabled": True,
        "use_geometry": True,
        "train_only": True,
        "strict": False,
        "missing": ["depth_residual_head."],
        "model_name": "qecr_v20b_stereo_geometry_depth_residual",
        "save_path": "outputs/第三阶段_V20立体几何深度残差/V20B_立体几何深度残差/",
    },
}

EXPECTED_AXIAL_DEPTH_IOU = {
    "V22O_原始Laplace深度控制": {
        "enabled": False,
        "model_name": "qecr_v22o_laplace_depth_control",
        "save_path": "outputs/第三阶段_V22轴向IoU深度损失/V22O_原始Laplace深度控制/",
    },
    "V22A_轴向IoU敏感深度损失": {
        "enabled": True,
        "model_name": "qecr_v22a_axial_iou_depth_loss",
        "save_path": "outputs/第三阶段_V22轴向IoU深度损失/V22A_轴向IoU敏感深度损失/",
    },
}

# These directories can be created by low-cost post-hoc diagnostics on the
# server.  They are not trainable entries in the formal switch matrix, but
# retaining them must not make the pre-training check fail.
OPTIONAL_DIAGNOSTIC_VERSIONS = {
    "V03A_融合权重置零评估",
}


def main():
    discovered = {}
    for name, expected in EXPECTED.items():
        path = ROOT / "versions" / name / "config.yaml"
        config = load_config(path)
        query = config["model"]["query_depth"]
        epipolar = config["model"]["query_epipolar"]
        observed = (
            bool(query["enabled"]),
            bool(query["cross_layer_refine"]),
            bool(query["baseline_safe_fusion"]),
            int(query["num_points"]),
            bool(epipolar["enabled"]),
            bool(epipolar["bidirectional"]),
            bool(epipolar["use_geometry_gate"]),
        )
        assert observed == expected, (name, observed, expected)
        assert not bool(query.get("offline_teacher", False))
        assert not bool(config["dataset"].get("teacher_cache", {}).get(
            "enabled", False
        ))
        assert not bool(query.get("geometry_gate", False))
        assert config["dataset"]["depth_scale"] == "normal"
        assert float(config["dataset"]["stereo_shift_aug"]) == 0.0
        assert int(config["model"]["dec_layers"]) == 3
        assert int(config["model"]["num_depth_bins"]) == 80
        if query["baseline_safe_fusion"]:
            assert float(query["initial_query_weight"]) == 0.05
            assert float(query["max_query_weight"]) == 0.50
        discovered[name] = path

    for name, expected in EXPECTED_DISPARITY.items():
        path = ROOT / "versions" / name / "config.yaml"
        config = load_config(path)
        dataset_supervision = config["dataset"]["disparity_supervision"]
        model_supervision = config["model"]["disparity_supervision"]
        observed = (
            str(config["dataset"]["matcher_mode"]),
            str(dataset_supervision["mode"]),
            bool(dataset_supervision["lrc_soft_weight"]),
        )
        assert observed == expected, (name, observed, expected)
        assert model_supervision["mode"] == dataset_supervision["mode"]
        assert bool(model_supervision["lrc_soft_weight"]) == bool(
            dataset_supervision["lrc_soft_weight"]
        )
        assert not bool(config["model"]["query_depth"]["enabled"])
        assert not bool(config["model"]["query_epipolar"]["enabled"])
        assert int(config["trainer"]["max_epoch"]) == 10
        assert float(config["optimizer"]["lr"]) == 0.00002
        assert config["trainer"].get("pretrain_model")
        discovered[name] = path

    for name, (pairwise_expected, guided_topk_expected) in EXPECTED_QUALITY.items():
        path = ROOT / "versions" / name / "config.yaml"
        config = load_config(path)
        quality = config["model"]["quality_ranking"]
        assert bool(quality["enabled"])
        assert bool(quality["freeze_detector"])
        assert bool(quality["quality_only_when_frozen"])
        assert bool(quality["pairwise_enabled"]) == pairwise_expected
        assert bool(config["tester"].get("quality_guided_topk", False)) == (
            guided_topk_expected
        )
        assert float(quality["point_loss_coef"]) == 10.0
        assert float(quality["negative_threshold"]) == 0.1
        assert float(quality["negative_weight"]) == 0.1
        if pairwise_expected:
            assert float(quality["pairwise_loss_coef"]) == 0.5
            assert float(quality["pairwise_margin"]) == 0.1
            assert int(quality["max_pairs_per_class"]) == 32
        dataset = config["dataset"]
        assert not bool(dataset["aug_crop"])
        assert float(dataset["random_mixup3d"]) == 0.0
        assert float(dataset["random_flip"]) == 0.0
        assert float(dataset["random_crop"]) == 0.0
        assert float(dataset["random_switch"]) == 0.0
        assert int(config["trainer"]["max_epoch"]) == 5
        assert float(config["optimizer"]["lr"]) == 0.001
        assert config["trainer"].get("pretrain_model")
        assert config["trainer"].get("pretrain_strict") is False
        if guided_topk_expected:
            assert not pairwise_expected
            assert float(quality["score_power"]) == 1.5
        discovered[name] = path

    for name, expected in EXPECTED_GEOMETRY.items():
        path = ROOT / "versions" / name / "config.yaml"
        config = load_config(path)
        geometry = config["model"]["geometry_alignment"]
        observed = (
            bool(geometry["enabled"]),
            bool(geometry["corner_enabled"]),
            bool(geometry["projection_enabled"]),
        )
        assert observed == expected, (name, observed, expected)
        assert float(geometry["corner_loss_coef"]) == 1.0
        assert float(geometry["projection_loss_coef"]) == 1.0
        assert float(geometry["projection_boundary_weight"]) == 0.25
        assert int(geometry["start_epoch"]) == 1
        assert int(geometry["ramp_epochs"]) == 4
        dataset = config["dataset"]
        assert not bool(dataset["aug_pd"])
        assert not bool(dataset["aug_crop"])
        assert float(dataset["random_mixup3d"]) == 0.0
        assert float(dataset["random_flip"]) == 0.0
        assert float(dataset["random_crop"]) == 0.0
        assert float(dataset["random_switch"]) == 0.0
        assert int(config["trainer"]["max_epoch"]) == 5
        assert float(config["optimizer"]["lr"]) == 0.00002
        assert config["trainer"].get("pretrain_model")
        assert config["trainer"].get("pretrain_strict") is True
        assert config["trainer"]["save_path"] == EXPECTED_GEOMETRY_OUTPUTS[name]
        discovered[name] = path

    for name, expected in EXPECTED_FINAL_COMBINATIONS.items():
        path = ROOT / "versions" / name / "config.yaml"
        config = load_config(path)
        quality = config["model"]["quality_ranking"]
        geometry = config["model"]["geometry_alignment"]
        dataset = config["dataset"]
        trainer = config["trainer"]

        assert bool(quality["enabled"])
        assert bool(quality["freeze_detector"])
        assert bool(quality["quality_only_when_frozen"])
        assert not bool(quality["pairwise_enabled"])
        assert float(quality["score_power"]) == 1.5
        assert float(quality["point_loss_coef"]) == 10.0
        assert float(quality["negative_threshold"]) == 0.1
        assert float(quality["negative_weight"]) == 0.1
        assert not bool(config["tester"].get("quality_guided_topk", False))

        assert not bool(geometry["enabled"])
        assert not bool(geometry["corner_enabled"])
        assert not bool(geometry["projection_enabled"])
        assert not bool(config["model"]["query_depth"]["enabled"])
        assert not bool(config["model"]["query_epipolar"]["enabled"])

        assert str(dataset["matcher_mode"]) == "StereoBM"
        assert not bool(dataset["aug_pd"])
        assert not bool(dataset["aug_crop"])
        assert float(dataset["random_mixup3d"]) == 0.0
        assert float(dataset["random_flip"]) == 0.0
        assert float(dataset["random_crop"]) == 0.0
        assert float(dataset["random_switch"]) == 0.0

        assert int(trainer["max_epoch"]) == 5
        assert float(config["optimizer"]["lr"]) == 0.001
        assert trainer["pretrain_model"] == expected["pretrain_model"]
        assert trainer["pretrain_strict"] is False
        assert trainer["pretrain_allowed_missing_prefixes"] == [
            "quality_head."
        ]
        assert trainer["pretrain_allow_unexpected"] is False
        assert trainer["save_path"] == expected["save_path"]
        assert not bool(config["model"]["depth_readout"]["enabled"])
        discovered[name] = path

    depth_readout_diagnostics = {}
    for name, expected in EXPECTED_DEPTH_READOUT_DIAGNOSTICS.items():
        path = ROOT / "versions" / name / "config.yaml"
        config = load_config(path)
        readout = config["model"]["depth_readout"]
        quality = config["model"]["quality_ranking"]
        trainer = config["trainer"]
        assert bool(readout["enabled"]) == expected["enabled"]
        assert int(readout["top_k"]) == expected["top_k"]
        assert config["model_name"] == expected["model_name"]
        assert bool(quality["enabled"])
        assert bool(quality["freeze_detector"])
        assert float(quality["score_power"]) == 1.5
        assert not bool(config["model"]["query_depth"]["enabled"])
        assert not bool(config["model"]["query_epipolar"]["enabled"])
        assert not bool(config["model"]["geometry_alignment"]["enabled"])
        assert trainer["pretrain_strict"] is True
        assert trainer["pretrain_allowed_missing_prefixes"] == []
        assert trainer["pretrain_allow_unexpected"] is False
        assert trainer["save_path"] == expected["save_path"]
        assert trainer["pretrain_model"].endswith(
            "qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
        )
        depth_readout_diagnostics[name] = path

    dynamic_upsampling_configs = {}
    for name, expected in EXPECTED_DYNAMIC_DEPTH_UPSAMPLING.items():
        path = ROOT / "versions" / name / "config.yaml"
        config = load_config(path)
        dynamic = config["model"]["dynamic_depth_upsampling"]
        quality = config["model"]["quality_ranking"]
        trainer = config["trainer"]
        optimizer = config["optimizer"]
        assert bool(dynamic["enabled"]) == expected["enabled"]
        assert list(dynamic["stages"]) == expected["stages"]
        assert float(dynamic["max_offset"]) == 0.5
        assert config["model"]["depth_map_size_mode"] == "downsample_x4"
        assert bool(dynamic["train_only_depth_classifier"])
        assert dynamic["trainable_prefixes"] == [
            "depth_predictor.depth_classifier."
        ]
        assert bool(quality["enabled"])
        assert not bool(quality["loss_enabled"])
        assert not bool(quality["freeze_detector"])
        assert float(quality["score_power"]) == 1.5
        assert config["model_name"] == expected["model_name"]
        assert int(trainer["max_epoch"]) == 5
        assert float(optimizer["lr"]) == 0.00002
        assert optimizer["parameter_lr_multipliers"] == {
            "offset_predictor": 10.0
        }
        assert trainer["pretrain_strict"] is expected["strict"]
        assert trainer["pretrain_allowed_missing_prefixes"] == expected[
            "missing"
        ]
        assert trainer["pretrain_allow_unexpected"] is False
        assert trainer["save_path"] == expected["save_path"]
        assert trainer["pretrain_model"].endswith(
            "qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
        )
        dynamic_upsampling_configs[name] = path

    groupwise_correlation_configs = {}
    for name, expected in EXPECTED_GROUPWISE_CORRELATION.items():
        path = ROOT / "versions" / name / "config.yaml"
        config = load_config(path)
        groupwise = config["model"]["groupwise_correlation"]
        quality = config["model"]["quality_ranking"]
        trainer = config["trainer"]
        optimizer = config["optimizer"]
        assert int(config.get("random_seed", 444)) == expected["seed"]
        assert bool(groupwise["enabled"]) == expected["enabled"]
        assert int(groupwise["scale"]) == 4
        assert int(groupwise["num_groups"]) == 16
        assert int(groupwise["gate_kernel_size"]) == 3
        assert bool(groupwise["train_only_cost_aggregation"])
        assert groupwise["trainable_prefixes"] == expected["prefixes"]
        assert bool(quality["enabled"])
        assert not bool(quality["loss_enabled"])
        assert not bool(quality["freeze_detector"])
        assert float(quality["score_power"]) == 1.5
        assert config["model_name"] == expected["model_name"]
        assert int(trainer["max_epoch"]) == 5
        assert float(optimizer["lr"]) == 0.00002
        assert optimizer["parameter_lr_multipliers"] == {
            "groupwise_correlation_s4": 10.0
        }
        assert trainer["pretrain_strict"] is expected["strict"]
        assert trainer["pretrain_allowed_missing_prefixes"] == expected[
            "missing"
        ]
        assert trainer["pretrain_allow_unexpected"] is False
        assert trainer["save_path"] == expected["save_path"]
        assert trainer["pretrain_model"].endswith(
            "qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
        )
        groupwise_correlation_configs[name] = path

    fixed_smoothing_diagnostics = {}
    for name, expected in EXPECTED_FIXED_SMOOTHING_DIAGNOSTICS.items():
        path = ROOT / "versions" / name / "config.yaml"
        config = load_config(path)
        smoothing = config["model"]["correlation_smoothing"]
        quality = config["model"]["quality_ranking"]
        trainer = config["trainer"]
        assert bool(smoothing["enabled"]) is expected["enabled"]
        assert int(smoothing["scale"]) == 4
        assert int(smoothing["kernel_size"]) == 3
        assert float(smoothing["blend"]) == 1.0
        assert int(smoothing["passes"]) == 1
        assert config["model_name"] == expected["model_name"]
        assert bool(quality["enabled"])
        assert float(quality["score_power"]) == 1.5
        assert not bool(config["model"]["groupwise_correlation"]["enabled"])
        assert trainer["save_path"] == expected["save_path"]
        fixed_smoothing_diagnostics[name] = path

    geometry_depth_residual_configs = {}
    for name, expected in EXPECTED_GEOMETRY_DEPTH_RESIDUAL.items():
        path = ROOT / "versions" / name / "config.yaml"
        config = load_config(path)
        residual = config["model"]["geometry_depth_residual"]
        quality = config["model"]["quality_ranking"]
        trainer = config["trainer"]
        assert bool(residual["enabled"]) is expected["enabled"]
        assert bool(residual["use_geometry_prior"]) is expected["use_geometry"]
        assert bool(residual["train_only_head"]) is expected["train_only"]
        assert int(residual["hidden_dim"]) == 32
        assert float(residual["max_correction"]) == 3.0
        assert bool(residual["detach_inputs"])
        assert residual["trainable_prefixes"] == ["depth_residual_head."]
        assert config["model_name"] == expected["model_name"]
        assert bool(quality["enabled"])
        assert float(quality["score_power"]) == 1.5
        assert trainer["pretrain_strict"] is expected["strict"]
        assert trainer["pretrain_allowed_missing_prefixes"] == expected["missing"]
        assert trainer["pretrain_allow_unexpected"] is False
        assert trainer["save_path"] == expected["save_path"]
        assert trainer["pretrain_model"].endswith(
            "qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
        )
        if expected["enabled"]:
            assert not bool(quality["loss_enabled"])
            assert not bool(quality["freeze_detector"])
            assert int(trainer["max_epoch"]) == 5
            assert float(config["optimizer"]["lr"]) == 0.001
        geometry_depth_residual_configs[name] = path

    axial_depth_iou_configs = {}
    for name, expected in EXPECTED_AXIAL_DEPTH_IOU.items():
        path = ROOT / "versions" / name / "config.yaml"
        config = load_config(path)
        axial = config["model"]["axial_depth_iou"]
        quality = config["model"]["quality_ranking"]
        trainer = config["trainer"]
        optimizer = config["optimizer"]
        assert bool(axial["enabled"]) is expected["enabled"]
        assert float(axial["loss_coef"]) == 0.5
        assert float(axial["min_half_extent"]) == 0.05
        assert float(axial["eps"]) == 0.000001
        assert bool(axial["train_only_depth_classifier"])
        assert axial["trainable_prefixes"] == [
            "depth_predictor.depth_classifier."
        ]
        assert config["model_name"] == expected["model_name"]
        assert bool(quality["enabled"])
        assert not bool(quality["loss_enabled"])
        assert not bool(quality["freeze_detector"])
        assert float(quality["score_power"]) == 1.5
        assert int(config.get("random_seed", 444)) == 444
        assert int(trainer["max_epoch"]) == 5
        assert float(optimizer["lr"]) == 0.00002
        assert optimizer["parameter_lr_multipliers"] == {}
        assert trainer["pretrain_strict"] is True
        assert trainer["pretrain_allowed_missing_prefixes"] == []
        assert trainer["pretrain_allow_unexpected"] is False
        assert trainer["save_path"] == expected["save_path"]
        assert trainer["pretrain_model"].endswith(
            "qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
        )
        axial_depth_iou_configs[name] = path

    control_pair = deepcopy(load_config(
        axial_depth_iou_configs["V22O_原始Laplace深度控制"]
    ))
    candidate_pair = deepcopy(load_config(
        axial_depth_iou_configs["V22A_轴向IoU敏感深度损失"]
    ))
    for paired in (control_pair, candidate_pair):
        paired.pop("_config_file", None)
        paired.pop("model_name")
        paired["trainer"].pop("save_path")
        paired["model"]["axial_depth_iou"].pop("enabled")
    assert control_pair == candidate_pair, (
        "V22O/V22A contain differences beyond the preregistered loss switch"
    )

    actual = {
        path.parent.name
        for path in (ROOT / "versions").glob("V*/config.yaml")
    }
    expected_names = (
        set(EXPECTED) | set(EXPECTED_DISPARITY) | set(EXPECTED_QUALITY)
        | set(EXPECTED_GEOMETRY) | set(EXPECTED_FINAL_COMBINATIONS)
        | set(EXPECTED_DEPTH_READOUT_DIAGNOSTICS)
        | set(EXPECTED_DYNAMIC_DEPTH_UPSAMPLING)
        | set(EXPECTED_GROUPWISE_CORRELATION)
        | set(EXPECTED_FIXED_SMOOTHING_DIAGNOSTICS)
        | set(EXPECTED_GEOMETRY_DEPTH_RESIDUAL)
        | set(EXPECTED_AXIAL_DEPTH_IOU)
    )
    missing = expected_names - actual
    unknown = actual - expected_names - OPTIONAL_DIAGNOSTIC_VERSIONS
    assert not missing, ("missing formal configs", sorted(missing))
    assert not unknown, ("unknown config directories", sorted(unknown))
    retained_diagnostics = actual & OPTIONAL_DIAGNOSTIC_VERSIONS
    smoke = load_config(ROOT / "configs" / "AutoDL_V06C_一轮冒烟.yaml")
    assert int(smoke["trainer"]["max_epoch"]) == 1
    assert bool(smoke["model"]["quality_ranking"]["pairwise_enabled"])
    geometry_smoke = load_config(
        ROOT / "configs" / "AutoDL_V08C_几何损失冒烟.yaml"
    )
    assert int(geometry_smoke["trainer"]["max_epoch"]) == 1
    assert bool(geometry_smoke["model"]["geometry_alignment"]["corner_enabled"])
    assert bool(geometry_smoke["model"]["geometry_alignment"]["projection_enabled"])
    assert int(geometry_smoke["model"]["geometry_alignment"]["start_epoch"]) == 0
    assert geometry_smoke["trainer"]["save_path"] == (
        "outputs/第二创新点_V08空间投影对齐/冒烟测试/"
    )
    dynamic_smoke = load_config(
        ROOT / "configs" / "AutoDL_V11A_动态上采样冒烟.yaml"
    )
    assert int(dynamic_smoke["trainer"]["max_epoch"]) == 1
    assert bool(
        dynamic_smoke["model"]["dynamic_depth_upsampling"]["enabled"]
    )
    assert dynamic_smoke["model"]["dynamic_depth_upsampling"]["stages"] == [1]
    assert dynamic_smoke["trainer"]["save_path"] == (
        "outputs/第二创新点_V11动态深度上采样/冒烟测试/"
    )
    groupwise_smoke = load_config(
        ROOT / "configs" / "AutoDL_V12A_分组相关门控冒烟.yaml"
    )
    assert int(groupwise_smoke["trainer"]["max_epoch"]) == 1
    assert bool(groupwise_smoke["model"]["groupwise_correlation"]["enabled"])
    assert int(
        groupwise_smoke["model"]["groupwise_correlation"]["num_groups"]
    ) == 16
    assert groupwise_smoke["trainer"]["save_path"] == (
        "outputs/第二创新点_V12分组相关门控/冒烟测试/"
    )
    residual_smoke = load_config(
        ROOT / "configs" / "AutoDL_V20B_立体几何深度残差冒烟.yaml"
    )
    assert int(residual_smoke["trainer"]["max_epoch"]) == 1
    assert bool(
        residual_smoke["model"]["geometry_depth_residual"]["enabled"]
    )
    assert bool(
        residual_smoke["model"]["geometry_depth_residual"][
            "use_geometry_prior"
        ]
    )
    assert residual_smoke["trainer"]["save_path"] == (
        "outputs/第三阶段_V20立体几何深度残差/冒烟测试/"
    )
    axial_smoke = load_config(
        ROOT / "configs" / "AutoDL_V22A_轴向IoU深度损失冒烟.yaml"
    )
    assert int(axial_smoke["trainer"]["max_epoch"]) == 1
    assert bool(axial_smoke["model"]["axial_depth_iou"]["enabled"])
    assert axial_smoke["trainer"]["save_path"] == (
        "outputs/第三阶段_V22轴向IoU深度损失/冒烟测试/"
    )
    print("{} 个无蒸馏正式消融配置检查通过".format(len(discovered)))
    print(
        "{} 个V10冻结权重深度读出诊断配置检查通过".format(
            len(depth_readout_diagnostics)
        )
    )
    print(
        "{} 个V11动态深度上采样配置检查通过".format(
            len(dynamic_upsampling_configs)
        )
    )
    print(
        "{} 个V12分组相关门控配置检查通过".format(
            len(groupwise_correlation_configs)
        )
    )
    print(
        "{} 个V14固定相关平滑零训练复评配置检查通过".format(
            len(fixed_smoothing_diagnostics)
        )
    )
    print(
        "{} 个V20立体几何深度残差配置检查通过".format(
            len(geometry_depth_residual_configs)
        )
    )
    print(
        "{} 个V22轴向IoU敏感深度损失配置检查通过".format(
            len(axial_depth_iou_configs)
        )
    )
    if retained_diagnostics:
        print(
            "保留并忽略诊断配置：{}".format(
                ", ".join(sorted(retained_diagnostics))
            )
        )


if __name__ == "__main__":
    main()
