"""Summarize the pre-registered V12 seed-444/445 paired replication."""

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
V09_CAR_MODERATE = 49.8210


def parse_official_r40(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {}
    for class_name, header in HEADERS.items():
        pattern = re.compile(
            header + r"(?:(?!AP_R40@)[\s\S])*?3d\s+AP:"
            r"([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)"
        )
        matches = pattern.findall(text)
        if not matches:
            raise ValueError("{} missing {} 3D AP_R40".format(path, class_name))
        result[class_name] = tuple(float(value) for value in matches[-1])
    return result


def subtract(candidate, control):
    return {
        name: tuple(a - b for a, b in zip(candidate[name], control[name]))
        for name in HEADERS
    }


def average_deltas(first, second):
    return {
        name: tuple(
            statistics.mean((first[name][index], second[name][index]))
            for index in range(3)
        )
        for name in HEADERS
    }


def decision(delta_444, delta_445, mean_delta, candidate_444, candidate_445):
    replication_guards = (
        delta_445["Car"][1] >= 0.10,
        mean_delta["Car"][1] >= 0.15,
        mean_delta["Car"][0] >= -0.10,
        mean_delta["Car"][2] >= -0.10,
        delta_445["Pedestrian"][1] >= -0.50,
        delta_445["Cyclist"][1] >= -0.50,
        mean_delta["Pedestrian"][1] >= -0.30,
        mean_delta["Cyclist"][1] >= -0.30,
        mean_delta["Pedestrian"][2] >= -0.75,
        mean_delta["Cyclist"][2] >= -0.75,
    )
    module_pass = all(replication_guards)
    candidate_mean = statistics.mean(
        (candidate_444["Car"][1], candidate_445["Car"][1])
    )
    final_pass = module_pass and candidate_mean >= V09_CAR_MODERATE
    return module_pass, final_pass, candidate_mean


def triplet(values):
    return "{:.4f} / {:.4f} / {:.4f}".format(*values)


def build_report(control_444, candidate_444, control_445, candidate_445):
    delta_444 = subtract(candidate_444, control_444)
    delta_445 = subtract(candidate_445, control_445)
    mean_delta = average_deltas(delta_444, delta_445)
    module_pass, final_pass, candidate_mean = decision(
        delta_444, delta_445, mean_delta, candidate_444, candidate_445
    )
    lines = [
        "# V12第二随机种子复核报告",
        "",
        "## 预注册结论",
        "",
        "- 模块配对增益复现：**{}**。".format("通过" if module_pass else "未通过"),
        "- 最终结果超过V09：**{}**。两个V12A的Car Moderate平均为 `{:.4f}`，V09参考值为 `{:.4f}`。".format(
            "通过" if final_pass else "未通过", candidate_mean, V09_CAR_MODERATE
        ),
        "- 本判定规则在种子445训练前固定；不得在看到结果后放宽小类别保护线。",
        "",
        "## 两个随机种子的正式KITTI 3D AP_R40",
        "",
        "| 种子/类别 | 控制组 E/M/H | V12A E/M/H | A−O E/M/H |",
        "|---|---:|---:|---:|",
    ]
    for seed, control, candidate, change in (
        (444, control_444, candidate_444, delta_444),
        (445, control_445, candidate_445, delta_445),
    ):
        for class_name in HEADERS:
            lines.append(
                "| {} / {} | {} | {} | {:+.4f} / {:+.4f} / {:+.4f} |".format(
                    seed,
                    class_name,
                    triplet(control[class_name]),
                    triplet(candidate[class_name]),
                    *change[class_name]
                )
            )
    lines.extend([
        "",
        "## 两种子配对差值平均",
        "",
        "| 类别 | E/M/H |",
        "|---|---:|",
    ])
    for class_name in HEADERS:
        values = mean_delta[class_name]
        lines.append(
            "| {} | {:+.4f} / {:+.4f} / {:+.4f} |".format(
                class_name, *values
            )
        )
    lines.extend([
        "",
        "通过条件包括：种子445 Car Moderate至少+0.10、两种子平均至少+0.15；两种子平均Car Easy/Hard不低于-0.10；Pedestrian/Cyclist Moderate和Hard同时受保护。",
        "",
    ])
    return "\n".join(lines)


def self_test():
    base = {
        "Car": (68.0, 49.5, 41.5),
        "Pedestrian": (32.0, 24.5, 20.0),
        "Cyclist": (42.0, 23.0, 21.0),
    }
    first = dict(base)
    second = dict(base)
    first["Car"] = (68.2, 49.7, 41.6)
    second["Car"] = (68.2, 49.7, 41.6)
    # Raise the absolute candidate reference only for this synthetic test.
    first["Car"] = (68.2, 50.0, 41.6)
    second["Car"] = (68.2, 50.0, 41.6)
    report = build_report(base, first, base, second)
    assert "模块配对增益复现：**通过**" in report
    assert "最终结果超过V09：**通过**" in report
    print("V12双随机种子复核与收紧保护门槛自检通过")


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "control_444", "candidate_444", "control_445", "candidate_445"
    ):
        parser.add_argument("--{}_log".format(name), type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = (
        args.control_444_log,
        args.candidate_444_log,
        args.control_445_log,
        args.candidate_445_log,
        args.output,
    )
    if not all(required):
        raise ValueError("four logs and output are required")
    report = build_report(
        parse_official_r40(args.control_444_log),
        parse_official_r40(args.candidate_444_log),
        parse_official_r40(args.control_445_log),
        parse_official_r40(args.candidate_445_log),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print("双种子报告：{}".format(args.output))


if __name__ == "__main__":
    main()
