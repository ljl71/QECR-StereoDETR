"""Summarize the preregistered V22O/V22A KITTI R40 pair."""

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
            if active is not None:
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
    protected = (
        car[0] >= -0.10
        and car[2] >= -0.10
        and pedestrian[1] >= -0.10
        and pedestrian[2] >= -0.10
        and cyclist[1] >= -0.10
        and cyclist[2] >= -0.10
    )
    if car[1] >= 0.20 and protected:
        return (
            "精度通过",
            "Car Moderate至少提升0.20 AP，且Car Easy/Hard及两个小类别的Moderate/Hard均通过保护线；进入第二随机种子复核。",
        )
    if car[1] >= 0.10 and protected:
        return (
            "谨慎",
            "出现正向趋势但未达到+0.20 AP正式门槛；只允许补第二随机种子，不运行195轮。",
        )
    return (
        "否决",
        "没有达到预注册精度收益或多类别保护门槛；停止V22，不运行195轮。",
    )


def triplet(values) -> str:
    return "{:.4f} / {:.4f} / {:.4f}".format(*values)


def build_report(control, candidate, paths) -> str:
    status, reason = decision(control, candidate)
    lines = [
        "# V22轴向IoU敏感深度损失配对结果",
        "",
        "## 自动结论",
        "",
        "- V22自动判定：**{}**。{}".format(status, reason),
        "- V22O与V22A从同一个V09 checkpoint、同一随机种子和同一depth_classifier训练范围出发。",
        "- V22A唯一变量是一项训练期轴向1D GIoU辅助损失；没有新增模型参数或推理分支。",
        "- 本表是KITTI验证集正式3D AP_R40；G7 Oracle不混入本表。",
        "",
        "## KITTI验证集3D AP_R40",
        "",
        "| 类别 | V22O E/M/H | V22A E/M/H | A−O E/M/H |",
        "|---|---:|---:|---:|",
    ]
    for class_name in CLASSES:
        change = delta(candidate[class_name], control[class_name])
        lines.append(
            "| {} | {} | {} | {:+.4f} / {:+.4f} / {:+.4f} |".format(
                class_name,
                triplet(control[class_name]),
                triplet(candidate[class_name]),
                *change,
            )
        )
    lines.extend([
        "",
        "## 审计路径",
        "",
        "- V22O日志：`{}`".format(paths["control"]),
        "- V22A日志：`{}`".format(paths["candidate"]),
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
    candidate = dict(control)
    candidate["Car"] = (68.0, 49.72, 41.6)
    assert decision(control, candidate)[0] == "精度通过"
    candidate["Car"] = (68.0, 49.62, 41.6)
    assert decision(control, candidate)[0] == "谨慎"
    candidate["Pedestrian"] = (35.0, 24.7, 21.0)
    assert decision(control, candidate)[0] == "否决"
    print("V22结果解析与预注册止损判定自检通过")


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
