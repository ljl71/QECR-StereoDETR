"""Summarize the frozen-checkpoint V10 foreground Top-K diagnostic."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


CLASSES = ("Car", "Pedestrian", "Cyclist")
HEADERS = {
    "Car": "Car AP_R40@0.70, 0.70, 0.70:",
    "Pedestrian": "Pedestrian AP_R40@0.50, 0.50, 0.50:",
    "Cyclist": "Cyclist AP_R40@0.50, 0.50, 0.50:",
}
AP_PATTERN = re.compile(
    r"3d\s+AP:\s*([-+0-9.]+),\s*([-+0-9.]+),\s*([-+0-9.]+)"
)
CONTROL_TARGET = 49.8210
CONTROL_TOLERANCE = 0.05


def parse_official_r40(text: str) -> Dict[str, Tuple[float, float, float]]:
    history: Dict[str, List[Tuple[float, float, float]]] = {
        name: [] for name in CLASSES
    }
    active = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        for class_name, header in HEADERS.items():
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
        raise ValueError("日志缺少正式R40结果：{}".format(", ".join(missing)))
    return {name: history[name][-1] for name in CLASSES}


def deltas(
    control: Dict[str, Tuple[float, float, float]],
    candidate: Dict[str, Tuple[float, float, float]],
) -> Dict[str, Tuple[float, float, float]]:
    return {
        name: tuple(
            candidate[name][index] - control[name][index]
            for index in range(3)
        )
        for name in CLASSES
    }


def passes_registered_gate(delta: Dict[str, Tuple[float, float, float]]) -> bool:
    return (
        delta["Car"][1] >= 0.15
        and delta["Car"][2] >= -0.10
        and delta["Pedestrian"][1] >= -0.10
        and delta["Cyclist"][1] >= -0.10
    )


def decide(
    control: Dict[str, Tuple[float, float, float]],
    topk2: Dict[str, Tuple[float, float, float]],
    topk4: Dict[str, Tuple[float, float, float]],
) -> Tuple[str, str]:
    if abs(control["Car"][1] - CONTROL_TARGET) > CONTROL_TOLERANCE:
        return (
            "口径异常",
            "控制组未在49.8210±0.05内恢复V09；不解释Top-K结果。",
        )
    delta2 = deltas(control, topk2)
    delta4 = deltas(control, topk4)
    if passes_registered_gate(delta2):
        return (
            "通过",
            "预注册主候选K=2达到精度门槛；可以进入正式模块与时延验证。",
        )
    if passes_registered_gate(delta4):
        return (
            "K=4待独立复核",
            "主候选K=2未通过，但机制敏感性K=4通过；不得直接包装为创新点。",
        )
    return (
        "否决",
        "K=2和K=4均未达到预注册门槛；停止该读出路线，不投入训练。",
    )


def triplet(values: Tuple[float, float, float]) -> str:
    return "{:.4f} / {:.4f} / {:.4f}".format(*values)


def build_report(
    control: Dict[str, Tuple[float, float, float]],
    topk2: Dict[str, Tuple[float, float, float]],
    topk4: Dict[str, Tuple[float, float, float]],
) -> str:
    status, reason = decide(control, topk2, topk4)
    result_sets = {
        "V10A K=2": topk2,
        "V10B K=4": topk4,
    }
    lines = [
        "# V10前景Top-K深度读出冻结权重诊断",
        "",
        "## 结论",
        "",
        "- 自动判定：**{}**。{}".format(status, reason),
        "- 主候选预先固定为CoEx默认风格的`K=2`；`K=4`只用于机制敏感性检查。",
        "- 三组使用同一个V09最佳checkpoint，不训练、不改变候选框架和质量头。",
        "",
        "## 正式KITTI 3D AP_R40",
        "",
        "| 类别 | V10O原始全分布 | V10A K=2 | K2−O | V10B K=4 | K4−O |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    delta2 = deltas(control, topk2)
    delta4 = deltas(control, topk4)
    for class_name in CLASSES:
        lines.append(
            "| {} | {} | {} | {:+.4f} / {:+.4f} / {:+.4f} | {} | {:+.4f} / {:+.4f} / {:+.4f} |".format(
                class_name,
                triplet(control[class_name]),
                triplet(result_sets["V10A K=2"][class_name]),
                *delta2[class_name],
                triplet(result_sets["V10B K=4"][class_name]),
                *delta4[class_name],
            )
        )
    lines.extend(
        [
            "",
            "## 预注册止损门槛",
            "",
            "正式通过要求K=2同时满足：Car Moderate `≥+0.15 AP`、Car Hard `≥-0.10 AP`、Pedestrian和Cyclist Moderate均`≥-0.10 AP`。K=4单独通过只触发一次独立复核，不直接进入论文。",
            "",
            "该诊断只检验冻结权重下的读出方式，不证明新的深度特征已经学成。未通过时不得通过重新扫描K、温度或门控阈值追逐验证集。",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    control = {
        "Car": (68.0, 49.8210, 41.8),
        "Pedestrian": (32.0, 24.6, 20.4),
        "Cyclist": (42.0, 22.9, 21.8),
    }
    passing = {
        "Car": (68.1, 50.0, 41.8),
        "Pedestrian": (32.0, 24.6, 20.4),
        "Cyclist": (42.0, 22.9, 21.8),
    }
    failing = {
        "Car": (68.0, 49.80, 41.8),
        "Pedestrian": (32.0, 24.6, 20.4),
        "Cyclist": (42.0, 22.9, 21.8),
    }
    assert decide(control, passing, failing)[0] == "通过"
    assert decide(control, failing, passing)[0] == "K=4待独立复核"
    assert decide(control, failing, failing)[0] == "否决"
    print("V10结果解析、主候选优先级与预注册止损判定自检通过")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_log", type=Path)
    parser.add_argument("--topk2_log", type=Path)
    parser.add_argument("--topk4_log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    for name in ("control_log", "topk2_log", "topk4_log", "output"):
        if getattr(args, name) is None:
            raise ValueError("--{} is required".format(name))
    control = parse_official_r40(
        args.control_log.read_text(encoding="utf-8", errors="replace")
    )
    topk2 = parse_official_r40(
        args.topk2_log.read_text(encoding="utf-8", errors="replace")
    )
    topk4 = parse_official_r40(
        args.topk4_log.read_text(encoding="utf-8", errors="replace")
    )
    report = build_report(control, topk2, topk4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print("结果报告：{}".format(args.output))


if __name__ == "__main__":
    main()

