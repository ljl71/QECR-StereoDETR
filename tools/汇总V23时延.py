"""Summarize alternating synchronized latency runs for V23O and V23B."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path


def parse_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    values = {}
    for name in ("parameters", "median_ms", "mean_ms", "p90_ms"):
        match = re.search(r"^{}:\s*([0-9.]+)\s*$".format(name), text, re.M)
        if match is None:
            raise ValueError("{} missing {}".format(path, name))
        values[name] = float(match.group(1))
    return values


def collect(latency_dir: Path):
    result = {"V23O": [], "V23B": []}
    for name in result:
        paths = sorted(latency_dir.glob("*{}*.log".format(name)))
        if len(paths) != 3:
            raise ValueError("{} expected 3 logs, found {}".format(name, len(paths)))
        result[name] = [(path, parse_log(path)) for path in paths]
    return result


def build_report(result):
    control = [item[1]["median_ms"] for item in result["V23O"]]
    candidate = [item[1]["median_ms"] for item in result["V23B"]]
    control_average = statistics.mean(control)
    candidate_average = statistics.mean(candidate)
    average_delta = 100.0 * (candidate_average / control_average - 1.0)
    robust_delta = 100.0 * (
        statistics.median(candidate) / statistics.median(control) - 1.0
    )
    control_parameters = int(result["V23O"][0][1]["parameters"])
    candidate_parameters = int(result["V23B"][0][1]["parameters"])
    parameter_delta = candidate_parameters - control_parameters
    parameter_ratio = 100.0 * parameter_delta / control_parameters
    status = (
        "通过"
        if average_delta <= 3.0 and parameter_ratio <= 0.5
        else "未通过"
    )
    lines = [
        "# V23同步模型前向时延与参数量判定",
        "",
        "- 自动结论：**{}**。三组Median平均增幅为`{:+.2f}%`，上限为`+3.00%`。".format(
            status, average_delta
        ),
        "- 三组Median的中位数增幅为`{:+.2f}%`。".format(robust_delta),
        "- 参数增量为`{:+d}`，相对增幅为`{:+.3f}%`，上限为`+0.500%`。".format(
            parameter_delta, parameter_ratio
        ),
        "",
        "| 版本 | 三组Median (ms) | 平均 (ms) | 三组中位数 (ms) |",
        "|---|---:|---:|---:|",
        "| V23O | {} | {:.4f} | {:.4f} |".format(
            " / ".join("{:.4f}".format(value) for value in control),
            control_average,
            statistics.median(control),
        ),
        "| V23B | {} | {:.4f} | {:.4f} |".format(
            " / ".join("{:.4f}".format(value) for value in candidate),
            candidate_average,
            statistics.median(candidate),
        ),
        "",
        "该测试只统计CUDA同步的batch=1模型前向，不含数据读取、结果写入和官方评估。",
        "",
    ]
    return "\n".join(lines)


def self_test() -> None:
    result = {
        "V23O": [
            (Path(str(i)), {"median_ms": value, "parameters": 18459785.0})
            for i, value in enumerate((25.0, 25.2, 24.8))
        ],
        "V23B": [
            (Path(str(i)), {"median_ms": value, "parameters": 18485369.0})
            for i, value in enumerate((25.3, 25.4, 25.2))
        ],
    }
    report = build_report(result)
    assert "**通过**" in report
    assert "`+25584`" in report
    print("V23三组交替时延、3%上限与0.5%参数上限自检通过")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.latency_dir is None or args.output is None:
        raise ValueError("latency_dir and output are required")
    report = build_report(collect(args.latency_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print("V23时延报告：{}".format(args.output))


if __name__ == "__main__":
    main()
