"""Summarize matched V11O/V11A/V11B KITTI R40 results and stop gates."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    active: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        for class_name, header in TARGET_HEADERS.items():
            if header in line:
                active = class_name
                break
        else:
            if active is None:
                continue
            match = AP_PATTERN.search(line)
            if match:
                history[active].append(
                    tuple(float(value) for value in match.groups())
                )
                active = None
    missing = [name for name, values in history.items() if not values]
    if missing:
        raise ValueError("日志缺少正式R40结果：{}".format(missing))
    # train_val.py最后会复评checkpoint_best；最后一组才是正式比较值。
    return {name: history[name][-1] for name in CLASSES}


def delta_triplet(experiment, control):
    return tuple(experiment[index] - control[index] for index in range(3))


def stage1_decision(control, stage1) -> Tuple[str, str]:
    car = delta_triplet(stage1["Car"], control["Car"])
    pedestrian = stage1["Pedestrian"][1] - control["Pedestrian"][1]
    cyclist = stage1["Cyclist"][1] - control["Cyclist"][1]
    if car[1] >= 0.15 and car[2] >= -0.10 and not (
        pedestrian < 0.0 and cyclist < 0.0
    ):
        return (
            "精度通过",
            "Car Moderate达到+0.15 AP，Hard未越过-0.10止损线，且两个小类别没有同时下降。",
        )
    if 0.05 <= car[1] < 0.15 and car[2] >= -0.10:
        return (
            "谨慎",
            "Car Moderate只有0.05—0.15 AP趋势；先复评或补种子，不运行V11B。",
        )
    return (
        "否决",
        "未达到预注册精度门槛；停止V11B和完整训练。",
    )


def format_triplet(values) -> str:
    return "{:.4f} / {:.4f} / {:.4f}".format(*values)


def build_report(control, stage1, stage2, paths) -> str:
    status, reason = stage1_decision(control, stage1)
    lines = [
        "# V11轻量动态深度上采样配对结果",
        "",
        "## 自动结论",
        "",
        "- V11A自动判定：**{}**。{}".format(status, reason),
        "- 本判定只覆盖精度门槛；通过后还必须完成同步模型前向时延，Median增幅不得超过3%。",
        "- 三组必须来自同一个V09 checkpoint、同一随机种子和同一训练设置。",
        "",
        "## 正式KITTI 3D AP_R40",
        "",
        "| 类别 | V11O E/M/H | V11A E/M/H | A-O E/M/H |",
        "|---|---:|---:|---:|",
    ]
    for class_name in CLASSES:
        delta = delta_triplet(stage1[class_name], control[class_name])
        lines.append(
            "| {} | {} | {} | {:+.4f} / {:+.4f} / {:+.4f} |".format(
                class_name,
                format_triplet(control[class_name]),
                format_triplet(stage1[class_name]),
                *delta,
            )
        )
    if stage2 is not None:
        lines.extend(
            [
                "",
                "## V11B两级动态上采样",
                "",
                "| 类别 | V11O E/M/H | V11B E/M/H | B-O E/M/H |",
                "|---|---:|---:|---:|",
            ]
        )
        for class_name in CLASSES:
            delta = delta_triplet(stage2[class_name], control[class_name])
            lines.append(
                "| {} | {} | {} | {:+.4f} / {:+.4f} / {:+.4f} |".format(
                    class_name,
                    format_triplet(control[class_name]),
                    format_triplet(stage2[class_name]),
                    *delta,
                )
            )
    lines.extend(
        [
            "",
            "## 审计路径",
            "",
            "- V11O日志：`{}`".format(paths["control"]),
            "- V11A日志：`{}`".format(paths["stage1"]),
        ]
    )
    if paths.get("stage2"):
        lines.append("- V11B日志：`{}`".format(paths["stage2"]))
    lines.append("")
    return "\n".join(lines)


def self_test() -> None:
    sample = """
Car AP_R40@0.70, 0.70, 0.70:
3d   AP:68.0000, 49.5000, 41.6000
Pedestrian AP_R40@0.50, 0.50, 0.50:
3d   AP:35.0000, 25.0000, 21.0000
Cyclist AP_R40@0.50, 0.50, 0.50:
3d   AP:42.0000, 23.0000, 22.0000
"""
    control = parse_official_r40(sample)
    improved = dict(control)
    improved["Car"] = (68.1, 49.66, 41.51)
    improved["Pedestrian"] = (35.0, 24.9, 21.0)
    improved["Cyclist"] = (42.0, 23.1, 22.0)
    assert stage1_decision(control, improved)[0] == "精度通过"
    weak = dict(control)
    weak["Car"] = (68.0, 49.56, 41.6)
    assert stage1_decision(control, weak)[0] == "谨慎"
    print("V11结果解析与预注册止损判定自检通过")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_log", type=Path)
    parser.add_argument("--stage1_log", type=Path)
    parser.add_argument("--stage2_log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    for name in ("control_log", "stage1_log", "output"):
        if getattr(args, name) is None:
            raise ValueError("--{} is required".format(name))
    control = parse_official_r40(
        args.control_log.read_text(encoding="utf-8", errors="replace")
    )
    stage1 = parse_official_r40(
        args.stage1_log.read_text(encoding="utf-8", errors="replace")
    )
    stage2 = None
    if args.stage2_log is not None and args.stage2_log.is_file():
        stage2 = parse_official_r40(
            args.stage2_log.read_text(encoding="utf-8", errors="replace")
        )
    report = build_report(
        control,
        stage1,
        stage2,
        {
            "control": args.control_log,
            "stage1": args.stage1_log,
            "stage2": args.stage2_log,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print("结果报告：{}".format(args.output))


if __name__ == "__main__":
    main()
