#!/usr/bin/env python3
"""Summarize zero-training V14O/V14S KITTI R40 evaluation."""

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
        matched_header = False
        for class_name, header in TARGET_HEADERS.items():
            if header in line:
                active = class_name
                matched_header = True
                break
        if matched_header or active is None:
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
    return {name: history[name][-1] for name in CLASSES}


def delta(candidate, control):
    return tuple(candidate[index] - control[index] for index in range(3))


def decision(control, candidate) -> Tuple[str, str]:
    car = delta(candidate["Car"], control["Car"])
    pedestrian = delta(candidate["Pedestrian"], control["Pedestrian"])
    cyclist = delta(candidate["Cyclist"], control["Cyclist"])
    geometry_guard = car[0] >= -0.10 and car[2] >= -0.10
    small_class_guard = all(
        value >= -0.20
        for value in (
            pedestrian[1], pedestrian[2], cyclist[1], cyclist[2]
        )
    )
    if car[1] >= 0.15 and geometry_guard and small_class_guard:
        return (
            "零训练精度通过",
            "Car Moderate至少+0.15 AP，Car Easy/Hard与两个小类别均未越过保护线；允许建立5轮共同续训配对。",
        )
    if 0.05 <= car[1] < 0.15 and geometry_guard and small_class_guard:
        return (
            "谨慎",
            "Car Moderate只有+0.05至+0.15 AP；先重复复评，不启动训练。",
        )
    return (
        "否决",
        "离线稀疏视差收益没有稳定传递到正式3D AP，停止固定平滑路线。",
    )


def format_triplet(values) -> str:
    return "{:.4f} / {:.4f} / {:.4f}".format(*values)


def build_report(control, candidate, paths) -> str:
    status, reason = decision(control, candidate)
    lines = [
        "# V14固定相关平滑零训练复评结果",
        "",
        "## 自动结论",
        "",
        "- V14S自动判定：**{}**。{}".format(status, reason),
        "- 两组使用同一个V09 checkpoint，没有训练、没有新增参数，只改变s4相关体是否执行单次3×3空间平均。",
        "- G4中的Velodyne稀疏视差没有用于本次推理或评估。",
        "",
        "## KITTI验证集3D AP_R40",
        "",
        "| 类别 | V14O E/M/H | V14S E/M/H | S−O E/M/H |",
        "|---|---:|---:|---:|",
    ]
    for class_name in CLASSES:
        change = delta(candidate[class_name], control[class_name])
        lines.append(
            "| {} | {} | {} | {:+.4f} / {:+.4f} / {:+.4f} |".format(
                class_name,
                format_triplet(control[class_name]),
                format_triplet(candidate[class_name]),
                *change,
            )
        )
    lines.extend([
        "",
        "## 审计路径",
        "",
        "- V14O日志：`{}`".format(paths["control"]),
        "- V14S日志：`{}`".format(paths["candidate"]),
        "",
    ])
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
    improved["Car"] = (68.0, 49.66, 41.6)
    assert decision(control, improved)[0] == "零训练精度通过"
    weak = dict(control)
    weak["Car"] = (68.0, 49.56, 41.6)
    assert decision(control, weak)[0] == "谨慎"
    harmed = dict(control)
    harmed["Car"] = (68.0, 49.7, 41.3)
    assert decision(control, harmed)[0] == "否决"
    print("V14结果解析与零训练止损判定自检通过")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_log", type=Path)
    parser.add_argument("--candidate_log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    for name in ("control_log", "candidate_log", "output"):
        if getattr(args, name) is None:
            raise ValueError("--{} is required".format(name))
    control = parse_official_r40(
        args.control_log.read_text(encoding="utf-8", errors="replace")
    )
    candidate = parse_official_r40(
        args.candidate_log.read_text(encoding="utf-8", errors="replace")
    )
    report = build_report(
        control,
        candidate,
        {"control": args.control_log, "candidate": args.candidate_log},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print("结果报告：{}".format(args.output))


if __name__ == "__main__":
    main()
