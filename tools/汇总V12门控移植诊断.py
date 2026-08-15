"""Summarize whether trained V12 gates improve the untouched V09 model."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path


HEADERS = {
    "Car": r"Car AP_R40@0\.70, 0\.70, 0\.70:",
    "Pedestrian": r"Pedestrian AP_R40@0\.50, 0\.50, 0\.50:",
    "Cyclist": r"Cyclist AP_R40@0\.50, 0\.50, 0\.50:",
}


def parse(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    result = {}
    for name, header in HEADERS.items():
        pattern = re.compile(
            header + r"(?:(?!AP_R40@)[\s\S])*?3d\s+AP:"
            r"([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)"
        )
        matches = pattern.findall(text)
        if not matches:
            raise ValueError("{} missing {} AP_R40".format(path, name))
        result[name] = tuple(float(value) for value in matches[-1])
    return result


def delta(candidate, control):
    return {
        name: tuple(a - b for a, b in zip(candidate[name], control[name]))
        for name in HEADERS
    }


def triplet(values):
    return "{:.4f} / {:.4f} / {:.4f}".format(*values)


def build_report(control, first, second):
    changes = [delta(first, control), delta(second, control)]
    mean = {
        name: tuple(
            statistics.mean((changes[0][name][i], changes[1][name][i]))
            for i in range(3)
        )
        for name in HEADERS
    }
    guards = (
        min(change["Car"][1] for change in changes) >= -0.05,
        mean["Car"][1] >= 0.10,
        mean["Car"][2] >= -0.10,
        mean["Pedestrian"][1] >= -0.30,
        mean["Cyclist"][1] >= -0.30,
        mean["Pedestrian"][2] >= -0.75,
        mean["Cyclist"][2] >= -0.75,
    )
    passed = all(guards)
    lines = [
        "# V12门控独立性移植诊断",
        "",
        "- 自动结论：**{}**。{}".format(
            "通过进入门控单独训练" if passed else "否决并停止V12",
            "两个已训练门控在逐位保持V09其余参数时仍有可利用增益。"
            if passed else
            "门控离开共同续训的cost_agg后没有满足预注册收益与多类别保护。"
        ),
        "- 本实验没有训练；只把三个门控张量移植进V09，其余V09张量逐位不变。",
        "",
        "| 模型/类别 | E/M/H | 相对V09 E/M/H |",
        "|---|---:|---:|",
    ]
    for label, values, change in (
        ("H1 seed444门控", first, changes[0]),
        ("H2 seed445门控", second, changes[1]),
    ):
        for name in HEADERS:
            lines.append(
                "| {} / {} | {} | {:+.4f} / {:+.4f} / {:+.4f} |".format(
                    label, name, triplet(values[name]), *change[name]
                )
            )
    lines.extend([
        "",
        "## 两个门控相对V09的平均差值",
        "",
        "| 类别 | E/M/H |",
        "|---|---:|",
    ])
    for name in HEADERS:
        lines.append(
            "| {} | {:+.4f} / {:+.4f} / {:+.4f} |".format(name, *mean[name])
        )
    lines.extend([
        "",
        "预注册条件：两个门控的Car Moderate均不得低于V09 0.05 AP以上，平均至少+0.10；同时保护Car Hard及Pedestrian/Cyclist Moderate、Hard。",
        "",
    ])
    return "\n".join(lines)


def self_test():
    control = {
        "Car": (69.0, 49.8, 41.8),
        "Pedestrian": (33.0, 24.6, 20.4),
        "Cyclist": (42.7, 22.9, 21.8),
    }
    first = {name: tuple(values) for name, values in control.items()}
    second = {name: tuple(values) for name, values in control.items()}
    first["Car"] = (69.0, 50.0, 41.85)
    second["Car"] = (69.0, 49.92, 41.82)
    report = build_report(control, first, second)
    assert "**通过进入门控单独训练**" in report
    print("V12门控独立性诊断与止损门槛自检通过")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_log", type=Path)
    parser.add_argument("--seed444_log", type=Path)
    parser.add_argument("--seed445_log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.control_log, args.seed444_log, args.seed445_log, args.output)):
        raise ValueError("three logs and output are required")
    report = build_report(
        parse(args.control_log), parse(args.seed444_log), parse(args.seed445_log)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print("移植诊断报告：{}".format(args.output))


if __name__ == "__main__":
    main()
