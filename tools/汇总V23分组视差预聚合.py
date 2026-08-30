"""Summarize V23O/V23A/V23B official KITTI R40 paired results."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CLASSES = ("Car", "Pedestrian", "Cyclist")
HEADERS = {
    "Car": r"Car AP_R40@0\.70, 0\.70, 0\.70:",
    "Pedestrian": r"Pedestrian AP_R40@0\.50, 0\.50, 0\.50:",
    "Cyclist": r"Cyclist AP_R40@0\.50, 0\.50, 0\.50:",
}
V09_CAR_MODERATE = 49.8210


def parse_official_r40(text: str):
    result = {}
    for class_name, header in HEADERS.items():
        pattern = re.compile(
            header + r"(?:(?!AP_R40@)[\s\S])*?3d\s+AP:"
            r"([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)"
        )
        matches = pattern.findall(text)
        if not matches:
            raise ValueError("missing {} 3D AP_R40".format(class_name))
        result[class_name] = tuple(float(value) for value in matches[-1])
    return result


def delta(candidate, control):
    return {
        name: tuple(a - b for a, b in zip(candidate[name], control[name]))
        for name in CLASSES
    }


def guards(change) -> bool:
    return (
        change["Car"][0] >= -0.10
        and change["Car"][2] >= -0.10
        and change["Pedestrian"][1] >= -0.20
        and change["Pedestrian"][2] >= -0.20
        and change["Cyclist"][1] >= -0.20
        and change["Cyclist"][2] >= -0.20
    )


def decision(control, rdsa, gpsd):
    a_o = delta(rdsa, control)
    b_o = delta(gpsd, control)
    b_a = delta(gpsd, rdsa)
    if b_o["Car"][1] >= 0.20 and b_a["Car"][1] >= 0.10 and guards(b_o):
        return (
            "精度通过",
            "GPSD相对公平控制的Car Moderate至少+0.20 AP，且相对RDSA至少+0.10 AP，多类别保护同时成立。",
        )
    if b_o["Car"][1] >= 0.20 and guards(b_o):
        return (
            "聚合有效但分组独立性不足",
            "GPSD达到总收益线，但没有比RDSA高0.10 AP。先补第二种子，暂不把收益归因于分组保真。",
        )
    if b_o["Car"][1] >= 0.10 and b_a["Car"][1] >= 0.0 and guards(b_o):
        return (
            "谨慎",
            "GPSD呈正增益但未达到0.20 AP稳健门槛。只允许补第二随机种子，不直接完整训练。",
        )
    return (
        "否决",
        "GPSD未达到预注册精度、独立贡献或多类别保护门槛，停止该路线。",
    )


def triplet(values) -> str:
    return "{:.4f} / {:.4f} / {:.4f}".format(*values)


def build_report(control, rdsa, gpsd, paths) -> str:
    status, reason = decision(control, rdsa, gpsd)
    a_o = delta(rdsa, control)
    b_o = delta(gpsd, control)
    b_a = delta(gpsd, rdsa)
    lines = [
        "# V23分组保真空间视差预聚合配对结果",
        "",
        "## 预注册结论",
        "",
        "- V23B自动判定：**{}**。{}".format(status, reason),
        "- V23O、V23A、V23B来自同一V09 checkpoint、同一随机种子和同一cost_agg训练范围。",
        "- V23A只增加666个3D微聚合参数。V23B只增加25584个GPSD参数。",
        "- V09参考Car Moderate为`{:.4f}`，当前V23B为`{:.4f}`。".format(
            V09_CAR_MODERATE, gpsd["Car"][1]
        ),
        "- 通过本轮仍只授权第二随机种子和同步CUDA时延复测。",
        "",
        "## KITTI验证集3D AP_R40",
        "",
        "| 类别 | V23O E/M/H | V23A E/M/H | A−O | V23B E/M/H | B−O | B−A |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CLASSES:
        lines.append(
            "| {} | {} | {} | {:+.4f}/{:+.4f}/{:+.4f} | {} | "
            "{:+.4f}/{:+.4f}/{:+.4f} | {:+.4f}/{:+.4f}/{:+.4f} |".format(
                name,
                triplet(control[name]),
                triplet(rdsa[name]),
                *a_o[name],
                triplet(gpsd[name]),
                *b_o[name],
                *b_a[name],
            )
        )
    lines.extend([
        "",
        "## 审计路径",
        "",
        "- V23O日志：`{}`".format(paths["control"]),
        "- V23A日志：`{}`".format(paths["rdsa"]),
        "- V23B日志：`{}`".format(paths["gpsd"]),
        "",
    ])
    return "\n".join(lines)


def self_test() -> None:
    base = {
        "Car": (68.0, 49.50, 41.50),
        "Pedestrian": (33.0, 24.5, 20.0),
        "Cyclist": (42.0, 23.0, 21.0),
    }
    rdsa = dict(base)
    rdsa["Car"] = (68.0, 49.58, 41.5)
    gpsd = dict(base)
    gpsd["Car"] = (68.0, 49.78, 41.5)
    assert decision(base, rdsa, gpsd)[0] == "精度通过"
    gpsd["Car"] = (68.0, 49.62, 41.5)
    assert decision(base, rdsa, gpsd)[0] == "谨慎"
    gpsd["Car"] = (68.0, 49.45, 41.5)
    assert decision(base, rdsa, gpsd)[0] == "否决"
    print("V23结果解析、RDSA独立对照与预注册止损判定自检通过")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_log", type=Path)
    parser.add_argument("--rdsa_log", type=Path)
    parser.add_argument("--gpsd_log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.control_log, args.rdsa_log, args.gpsd_log, args.output)):
        raise ValueError("three logs and output are required")
    control = parse_official_r40(
        args.control_log.read_text(encoding="utf-8", errors="replace")
    )
    rdsa = parse_official_r40(
        args.rdsa_log.read_text(encoding="utf-8", errors="replace")
    )
    gpsd = parse_official_r40(
        args.gpsd_log.read_text(encoding="utf-8", errors="replace")
    )
    report = build_report(
        control,
        rdsa,
        gpsd,
        {
            "control": args.control_log,
            "rdsa": args.rdsa_log,
            "gpsd": args.gpsd_log,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print("V23配对报告：{}".format(args.output))


if __name__ == "__main__":
    main()
