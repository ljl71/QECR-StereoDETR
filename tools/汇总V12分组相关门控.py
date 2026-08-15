"""Summarize the paired V12O/V12A official KITTI evaluation results."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HEADERS = {
    "Car": r"Car AP_R40@0\.70, 0\.70, 0\.70:",
    "Pedestrian": r"Pedestrian AP_R40@0\.50, 0\.50, 0\.50:",
    "Cyclist": r"Cyclist AP_R40@0\.50, 0\.50, 0\.50:",
}


def parse_official_r40(text: str):
    result = {}
    for class_name, header in HEADERS.items():
        pattern = re.compile(
            header + r"(?:(?!AP_R40@)[\s\S])*?3d\s+AP:"
            r"([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)"
        )
        matches = pattern.findall(text)
        if not matches:
            raise ValueError("没有找到{}正式3D AP_R40".format(class_name))
        result[class_name] = tuple(float(value) for value in matches[-1])
    return result


def delta(candidate, control):
    return tuple(a - b for a, b in zip(candidate, control))


def decision(control, candidate):
    car = delta(candidate["Car"], control["Car"])
    ped = delta(candidate["Pedestrian"], control["Pedestrian"])
    cyc = delta(candidate["Cyclist"], control["Cyclist"])
    guards = (
        car[0] >= -0.20,
        car[2] >= -0.10,
        ped[1] >= -0.50,
        cyc[1] >= -0.50,
    )
    if car[1] >= 0.15 and all(guards):
        return (
            "精度通过",
            "Car Moderate提升至少0.15 AP且Easy/Hard及小类别未触发退化门槛；"
            "下一步做时延与第二随机种子复核。",
        )
    if car[1] >= 0.05 and all(guards):
        return (
            "谨慎",
            "出现小幅正增益但尚未超过0.15 AP稳健门槛；先补第二随机种子，"
            "不直接完整训练。",
        )
    return (
        "否决",
        "未达到预注册精度门槛；停止V12完整训练并保留G0为负结果诊断。",
    )


def triplet(values):
    return "{:.4f} / {:.4f} / {:.4f}".format(*values)


def build_report(control, candidate, control_log, candidate_log):
    status, reason = decision(control, candidate)
    lines = [
        "# V12轻量分组相关门控配对结果",
        "",
        "## 自动结论",
        "",
        "- V12A自动判定：**{}**。{}".format(status, reason),
        "- 两组从同一个V09 checkpoint、同一随机种子、同一cost_agg训练范围出发；"
        "V12A只额外增加416个门控参数。",
        "- G0 Oracle只用于证明信息上限，本表才是可作为模型实验依据的正式验证集结果。",
        "",
        "## KITTI验证集3D AP_R40",
        "",
        "| 类别 | V12O E/M/H | V12A E/M/H | A-O E/M/H |",
        "|---|---:|---:|---:|",
    ]
    for class_name in ("Car", "Pedestrian", "Cyclist"):
        change = delta(candidate[class_name], control[class_name])
        lines.append(
            "| {} | {} | {} | {:+.4f} / {:+.4f} / {:+.4f} |".format(
                class_name,
                triplet(control[class_name]),
                triplet(candidate[class_name]),
                *change,
            )
        )
    lines.extend(
        [
            "",
            "## 审计路径",
            "",
            "- V12O日志：`{}`".format(control_log),
            "- V12A日志：`{}`".format(candidate_log),
            "",
        ]
    )
    return "\n".join(lines)


def self_test():
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
    improved["Car"] = (68.05, 49.66, 41.55)
    assert decision(control, improved)[0] == "精度通过"
    weak = dict(control)
    weak["Car"] = (68.0, 49.56, 41.6)
    assert decision(control, weak)[0] == "谨慎"
    failed = dict(control)
    failed["Car"] = (68.0, 49.4, 41.6)
    assert decision(control, failed)[0] == "否决"
    print("V12结果解析与预注册止损判定自检通过")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_log", type=Path)
    parser.add_argument("--candidate_log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.control_log, args.candidate_log, args.output)):
        raise ValueError("control_log, candidate_log and output are required")
    control = parse_official_r40(
        args.control_log.read_text(encoding="utf-8", errors="replace")
    )
    candidate = parse_official_r40(
        args.candidate_log.read_text(encoding="utf-8", errors="replace")
    )
    report = build_report(control, candidate, args.control_log, args.candidate_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print("结果报告：{}".format(args.output))


if __name__ == "__main__":
    main()
