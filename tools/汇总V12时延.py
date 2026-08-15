"""Summarize the three alternating V12O/V12A synchronized latency runs."""

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
    result = {"V12O": [], "V12A": []}
    for name in result:
        paths = sorted(latency_dir.glob("*{}*.log".format(name)))
        if len(paths) != 3:
            raise ValueError("{} expected 3 logs, found {}".format(name, len(paths)))
        result[name] = [(path, parse_log(path)) for path in paths]
    return result


def build_report(result):
    control = [item[1]["median_ms"] for item in result["V12O"]]
    candidate = [item[1]["median_ms"] for item in result["V12A"]]
    control_average = statistics.mean(control)
    candidate_average = statistics.mean(candidate)
    average_delta = 100.0 * (candidate_average / control_average - 1.0)
    robust_delta = 100.0 * (
        statistics.median(candidate) / statistics.median(control) - 1.0
    )
    status = "通过" if average_delta <= 3.0 else "未通过"
    control_parameters = int(result["V12O"][0][1]["parameters"])
    candidate_parameters = int(result["V12A"][0][1]["parameters"])
    lines = [
        "# V12同步模型前向时延判定",
        "",
        "- 自动结论：**{}**。三组 Median 平均增幅为 `{:+.2f}%`，预注册上限为 `+3.00%`。".format(
            status, average_delta
        ),
        "- 三组 Median 的中位数增幅为 `{:+.2f}%`，作为抗单组波动的辅助指标。".format(
            robust_delta
        ),
        "- 参数增量：`{:+d}`。".format(candidate_parameters - control_parameters),
        "",
        "| 版本 | 三组 Median (ms) | Median平均 (ms) | 三组中位数 (ms) |",
        "|---|---:|---:|---:|",
        "| V12O | {} | {:.4f} | {:.4f} |".format(
            " / ".join("{:.4f}".format(value) for value in control),
            control_average,
            statistics.median(control),
        ),
        "| V12A | {} | {:.4f} | {:.4f} |".format(
            " / ".join("{:.4f}".format(value) for value in candidate),
            candidate_average,
            statistics.median(candidate),
        ),
        "",
        "该测试只统计 CUDA 同步的 batch=1 模型前向，不包含数据读取、KITTI文本写入和官方评估。",
    ]
    return "\n".join(lines) + "\n"


def self_test():
    result = {
        "V12O": [(Path(str(i)), {"median_ms": value, "parameters": 100.0}) for i, value in enumerate((25.0, 25.2, 24.8))],
        "V12A": [(Path(str(i)), {"median_ms": value, "parameters": 516.0}) for i, value in enumerate((25.3, 25.4, 25.2))],
    }
    report = build_report(result)
    assert "**通过**" in report
    assert "`+416`" in report
    print("V12时延汇总与3%止损判定自检通过")


def main():
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
    print("时延报告：{}".format(args.output))


if __name__ == "__main__":
    main()
