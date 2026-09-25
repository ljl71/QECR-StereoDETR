#!/usr/bin/env python3
"""Summarize alternating synchronized latency runs for MQD variants."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path
from typing import Dict, List


FIELDS = ("parameters", "median_ms", "mean_ms", "p90_ms", "fps_from_median")
ADDED_QUERY_MACS = {
    "baseline": 0,
    "shared": 3_289_600,
    "complete": 3_700_800,
}


def parse_log(path: Path) -> Dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {}
    for field in FIELDS:
        match = re.search(r"^{}:\s*([-+0-9.eE]+)\s*$".format(field), text, re.M)
        if not match:
            raise ValueError("{} misses {}".format(path, field))
        result[field] = float(match.group(1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = ("baseline", "shared", "complete")
    values: Dict[str, List[Dict[str, float]]] = {}
    for variant in variants:
        logs = sorted(args.root.glob("*_{}.log".format(variant)))
        if len(logs) != 6:
            raise ValueError("{} requires six runs, got {}".format(variant, len(logs)))
        values[variant] = [parse_log(path) for path in logs]

    lines = [
        "# MQD完整模型同步前向时延",
        "",
        "- batch size为1，所有计时均包含CUDA同步。",
        "- 三个结构在同一GPU上交替测试六组。",
        "",
        "| 版本 | 参数量 | 新增查询MACs | 六组Median均值(ms) | Median的中位数(ms) | Mean均值(ms) | P90均值(ms) | FPS均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_median = statistics.mean(
        item["median_ms"] for item in values["baseline"]
    )
    for variant in variants:
        rows = values[variant]
        lines.append("| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |".format(
            variant,
            int(rows[0]["parameters"]),
            ADDED_QUERY_MACS[variant],
            statistics.mean(item["median_ms"] for item in rows),
            statistics.median(item["median_ms"] for item in rows),
            statistics.mean(item["mean_ms"] for item in rows),
            statistics.mean(item["p90_ms"] for item in rows),
            statistics.mean(item["fps_from_median"] for item in rows),
        ))
    complete_median = statistics.mean(
        item["median_ms"] for item in values["complete"]
    )
    lines.extend([
        "",
        "完整MQD相对基线的Median平均增幅：{:+.2f}%。".format(
            (complete_median / baseline_median - 1.0) * 100.0
        ),
        "",
        "新增MACs只统计QCMR与HPMD类别条件响应的查询级线性映射，不与论文中跨环境报告的整网FLOPs混算。",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
