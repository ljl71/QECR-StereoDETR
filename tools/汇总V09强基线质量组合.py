"""Summarize the matched V08O versus V09 KITTI evaluation logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


CLASSES = ("Car", "Pedestrian", "Cyclist")
TARGET_HEADERS = {
    "Car": "Car AP_R40@0.70, 0.70, 0.70:",
    "Pedestrian": "Pedestrian AP_R40@0.50, 0.50, 0.50:",
    "Cyclist": "Cyclist AP_R40@0.50, 0.50, 0.50:",
}
AP_PATTERN = re.compile(
    r"3d\s+AP:\s*([-+0-9.]+),\s*([-+0-9.]+),\s*([-+0-9.]+)"
)


def parse_official_r40(text: str) -> Dict[str, Tuple[float, float, float]]:
    history: Dict[str, List[Tuple[float, float, float]]] = {
        name: [] for name in CLASSES
    }
    active = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        matched_header = None
        for class_name, header in TARGET_HEADERS.items():
            if header in line:
                matched_header = class_name
                break
        if matched_header is not None:
            active = matched_header
            continue
        if active is None:
            continue
        match = AP_PATTERN.search(line)
        if match:
            history[active].append(tuple(float(value) for value in match.groups()))
            active = None

    missing = [name for name, values in history.items() if not values]
    if missing:
        raise ValueError("日志缺少正式R40结果：{}".format(", ".join(missing)))
    return {name: history[name][-1] for name in CLASSES}


def decision(
    control: Dict[str, Tuple[float, float, float]],
    experiment: Dict[str, Tuple[float, float, float]],
) -> Tuple[str, str]:
    if abs(control["Car"][1] - 49.5345) > 0.05:
        return (
            "口径异常",
            "R0未复现V08O历史49.5345（容差0.05 AP）；先检查checkpoint和配置，不解释R1。",
        )
    car_delta = tuple(
        experiment["Car"][index] - control["Car"][index]
        for index in range(3)
    )
    pedestrian_delta = experiment["Pedestrian"][1] - control["Pedestrian"][1]
    cyclist_delta = experiment["Cyclist"][1] - control["Cyclist"][1]
    if car_delta[1] >= 0.10 and car_delta[2] >= -0.10:
        return "通过", "Car Moderate达到+0.10 AP门槛且Hard未越过止损线。"
    if (
        car_delta[1] > 0.0
        and car_delta[2] >= -0.10
        and pedestrian_delta > 0.0
        and cyclist_delta > 0.0
    ):
        return (
            "多类别折中保留",
            "Car主指标不足+0.10 AP，但三类Moderate同步改善；不得宣称主指标显著提升。",
        )
    return "否决", "未达到预注册精度门槛，停止继续训练或扫描beta。"


def format_triplet(values: Tuple[float, float, float]) -> str:
    return "{:.4f} / {:.4f} / {:.4f}".format(*values)


def build_report(
    control: Dict[str, Tuple[float, float, float]],
    experiment: Dict[str, Tuple[float, float, float]],
    control_log: Path,
    experiment_log: Path,
) -> str:
    status, reason = decision(control, experiment)
    lines = [
        "# V09强基线加三维质量排序结果对照",
        "",
        "## 结论",
        "",
        "- 自动判定：**{}**。{}".format(status, reason),
        "- 直接分母：同一个V08O最佳checkpoint的R0复评；不得换用V00或早期独立复现。",
        "- R1只新增冻结检测器上的V06B点式质量头，推理固定beta=1.5。",
        "",
        "## 正式KITTI 3D AP_R40",
        "",
        "| 类别 | R0 V08O E/M/H | R1 V09 E/M/H | R1-R0 E/M/H |",
        "|---|---:|---:|---:|",
    ]
    for class_name in CLASSES:
        delta = tuple(
            experiment[class_name][index] - control[class_name][index]
            for index in range(3)
        )
        lines.append(
            "| {} | {} | {} | {:+.4f} / {:+.4f} / {:+.4f} |".format(
                class_name,
                format_triplet(control[class_name]),
                format_triplet(experiment[class_name]),
                *delta,
            )
        )
    lines.extend(
        [
            "",
            "## 口径检查",
            "",
            "- R0日志：`{}`".format(control_log),
            "- R1日志：`{}`".format(experiment_log),
            "- V08O历史目标Car Moderate：49.5345。若本次R0偏差超过0.05 AP，应先检查权重和配置，不解释R1。",
            "- beta=1.5已在先前G2诊断中固定；本实验完成后不根据结果重新搜索指数。",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    sample = """
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:1, 1, 1
3d   AP:68.0000, 49.5000, 41.6000
Car AP_R40@0.50, 0.50, 0.50:
3d   AP:90.0000, 80.0000, 70.0000
Pedestrian AP_R40@0.50, 0.50, 0.50:
3d   AP:35.0000, 25.0000, 21.0000
Pedestrian AP_R40@0.50, 0.25, 0.25:
3d   AP:70.0000, 60.0000, 50.0000
Cyclist AP_R40@0.50, 0.50, 0.50:
3d   AP:42.0000, 23.0000, 22.0000
Cyclist AP_R40@0.50, 0.25, 0.25:
3d   AP:60.0000, 40.0000, 35.0000
"""
    parsed = parse_official_r40(sample)
    assert parsed["Car"] == (68.0, 49.5, 41.6)
    assert parsed["Pedestrian"] == (35.0, 25.0, 21.0)
    assert parsed["Cyclist"] == (42.0, 23.0, 22.0)
    improved = dict(parsed)
    improved["Car"] = (68.1, 49.61, 41.55)
    assert decision(parsed, improved)[0] == "通过"
    print("V09结果解析与止损判定自检通过")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0_log", type=Path)
    parser.add_argument("--r1_log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    for name in ("r0_log", "r1_log", "output"):
        if getattr(args, name) is None:
            raise ValueError("--{} is required".format(name))
    control = parse_official_r40(
        args.r0_log.read_text(encoding="utf-8", errors="replace")
    )
    experiment = parse_official_r40(
        args.r1_log.read_text(encoding="utf-8", errors="replace")
    )
    report = build_report(
        control, experiment, args.r0_log, args.r1_log
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print("结果报告：{}".format(args.output))


if __name__ == "__main__":
    main()
