"""Load every ablation config and verify the intended feature matrix."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402


EXPECTED = {
    "V00_官方基线": (False, False, False, 1, False, False),
    "V01_查询级局部分布": (True, False, False, 1, False, False),
    "V02_跨层分布细化": (True, True, False, 1, False, False),
    "V03_质量对齐分类": (False, False, False, 1, True, False),
    "V04_不确定性几何门控": (True, False, True, 1, False, False),
    "V05_查询多点采样": (True, False, False, 5, False, False),
    "V06_离线教师蒸馏": (True, False, False, 1, False, True),
    "V06C_查询分布无MixUp对照": (True, False, False, 1, False, False),
    "V10_分布加质量对齐": (True, False, False, 1, True, False),
    "V11_分布细化加质量对齐": (True, True, False, 1, True, False),
    "V12_加入不确定性几何门控": (True, True, True, 1, True, False),
    "V13_加入查询多点采样": (True, True, True, 5, True, False),
    "V13C_完整结构无MixUp对照": (True, True, True, 5, True, False),
    "V14_完整模型": (True, True, True, 5, True, True),
}


def main() -> None:
    version_root = ROOT / "versions"
    for version_name, expected in EXPECTED.items():
        config_path = version_root / version_name / "config.yaml"
        config = load_config(config_path)
        query = config["model"]["query_depth"]
        observed = (
            query["enabled"],
            query["cross_layer_refine"],
            query["geometry_gate"],
            query["num_points"],
            query["quality_aligned_cls"],
            query["offline_teacher"],
        )
        assert observed == expected, (version_name, observed, expected)
        assert config["model"]["dec_layers"] == 3
        assert config["model"]["num_depth_bins"] == 80
        if query["offline_teacher"]:
            assert config["dataset"]["teacher_cache"]["enabled"]
            assert config["dataset"]["random_mixup3d"] == 0.0
            assert config["dataset"]["random_switch"] == 0.0
        if version_name in {
            "V06C_查询分布无MixUp对照", "V13C_完整结构无MixUp对照"
        }:
            assert config["dataset"]["random_mixup3d"] == 0.0
    discovered = {
        path.parent.name
        for path in version_root.glob("V*/config.yaml")
    }
    assert discovered == set(EXPECTED), (
        "unexpected or missing version configs",
        sorted(discovered),
    )
    print(f"{len(EXPECTED)} 个消融配置继承与开关矩阵检查通过")


if __name__ == "__main__":
    main()
