"""Summarize V20O/V20A/V20B KITTI R40 results with preregistered gates."""

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


def delta(experiment, control):
    return tuple(experiment[index] - control[index] for index in range(3))


def decision(control, query_only, geometry) -> Tuple[str, str]:
    overall = delta(geometry["Car"], control["Car"])
    geometry_gain = delta(geometry["Car"], query_only["Car"])
    pedestrian = delta(geometry["Pedestrian"], control["Pedestrian"])
    cyclist = delta(geometry["Cyclist"], control["Cyclist"])
    geometry_guard = overall[0] >= -0.10 and overall[2] >= -0.10
    small_class_guard = all(
        value >= -0.25
        for value in (
            pedestrian[1], pedestrian[2], cyclist[1], cyclist[2]
        )
    )
    if (
        overall[1] >= 0.20
        and geometry_gain[1] >= 0.10
        and geometry_guard
        and small_class_guard
    ):
        return (
            "精度通过",
            "V20B相对V09的Car Moderate至少+0.20 AP，且相对同参数量V20A至少+0.10 AP；几何先验贡献和多类别保护同时成立。",
        )
    if (
        overall[1] >= 0.15
        and geometry_gain[1] >= 0.0
        and geometry_guard
        and small_class_guard
    ):
        return (
            "谨慎",
            "总收益达到趋势线，但几何先验的独立贡献不足+0.10 AP；先做第二随机种子，不写成已验证创新点。",
        )
    if query_only["Car"][1] - control["Car"][1] >= 0.20:
        return (
            "只支持查询残差，不支持几何先验",
            "小型查询残差头有效，但V20B没有证明几何差值带来额外收益；只能保留V20A并重新定义贡献。",
        )
    return (
        "否决",
        "未达到预注册精度或多类别保护门槛；停止V20，不跑195轮。",
    )


def format_triplet(values) -> str:
    return "{:.4f} / {:.4f} / {:.4f}".format(*values)


def build_report(control, query_only, geometry, paths) -> str:
    status, reason = decision(control, query_only, geometry)
    lines = [
        "# V20立体几何深度残差配对结果",
        "",
        "## 自动结论",
        "",
        "- V20自动判定：**{}**。{}".format(status, reason),
        "- V20O是V09同权重零训练复评；V20A/V20B都冻结V09，只训练同参数量残差头。",
        "- V20A的几何输入恒为0；V20B唯一新增信息是现有几何深度与立体深度的归一化差值。",
        "- 通过精度门槛后仍必须做第二随机种子与CUDA同步时延复测。",
        "",
        "## KITTI验证集3D AP_R40",
        "",
        "| 类别 | V20O E/M/H | V20A E/M/H | A−O | V20B E/M/H | B−O | B−A |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for class_name in CLASSES:
        a_o = delta(query_only[class_name], control[class_name])
        b_o = delta(geometry[class_name], control[class_name])
        b_a = delta(geometry[class_name], query_only[class_name])
        lines.append(
            "| {} | {} | {} | {:+.4f}/{:+.4f}/{:+.4f} | {} | "
            "{:+.4f}/{:+.4f}/{:+.4f} | {:+.4f}/{:+.4f}/{:+.4f} |".format(
                class_name,
                format_triplet(control[class_name]),
                format_triplet(query_only[class_name]),
                *a_o,
                format_triplet(geometry[class_name]),
                *b_o,
                *b_a,
            )
        )
    lines.extend([
        "",
        "## 审计路径",
        "",
        "- V20O日志：`{}`".format(paths["control"]),
        "- V20A日志：`{}`".format(paths["query"]),
        "- V20B日志：`{}`".format(paths["geometry"]),
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
    query = dict(control)
    query["Car"] = (68.0, 49.58, 41.6)
    geometry = dict(control)
    geometry["Car"] = (68.0, 49.75, 41.6)
    assert decision(control, query, geometry)[0] == "精度通过"
    geometry["Car"] = (68.0, 49.66, 41.6)
    assert decision(control, query, geometry)[0] == "谨慎"
    geometry["Car"] = (68.0, 49.4, 41.6)
    assert decision(control, query, geometry)[0] == "否决"
    print("V20结果解析、几何独立贡献和预注册止损判定自检通过")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_log", type=Path)
    parser.add_argument("--query_log", type=Path)
    parser.add_argument("--geometry_log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    for name in ("control_log", "query_log", "geometry_log", "output"):
        if getattr(args, name) is None:
            raise ValueError("--{} is required".format(name))
    control = parse_official_r40(
        args.control_log.read_text(encoding="utf-8", errors="replace")
    )
    query = parse_official_r40(
        args.query_log.read_text(encoding="utf-8", errors="replace")
    )
    geometry = parse_official_r40(
        args.geometry_log.read_text(encoding="utf-8", errors="replace")
    )
    report = build_report(
        control,
        query,
        geometry,
        {
            "control": args.control_log,
            "query": args.query_log,
            "geometry": args.geometry_log,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print("结果报告：{}".format(args.output))


if __name__ == "__main__":
    main()

