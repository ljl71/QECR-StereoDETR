"""Summarize V23 seed-444/445 replication with fixed decision gates."""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "汇总V23分组视差预聚合.py"
SPEC = importlib.util.spec_from_file_location("v23_pair_summary", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)
PAIR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PAIR
SPEC.loader.exec_module(PAIR)


def average_changes(first, second):
    return {
        name: tuple(
            statistics.mean((first[name][index], second[name][index]))
            for index in range(3)
        )
        for name in PAIR.CLASSES
    }


def decision(o444, a444, b444, o445, a445, b445):
    bo444 = PAIR.delta(b444, o444)
    bo445 = PAIR.delta(b445, o445)
    ba444 = PAIR.delta(b444, a444)
    ba445 = PAIR.delta(b445, a445)
    mean_bo = average_changes(bo444, bo445)
    mean_ba = average_changes(ba444, ba445)
    replicate = (
        bo445["Car"][1] >= 0.10
        and mean_bo["Car"][1] >= 0.20
        and mean_ba["Car"][1] >= 0.10
        and PAIR.guards(mean_bo)
    )
    absolute_mean = statistics.mean((b444["Car"][1], b445["Car"][1]))
    final_pass = replicate and absolute_mean >= PAIR.V09_CAR_MODERATE
    return replicate, final_pass, absolute_mean, mean_bo, mean_ba


def build_report(o444, a444, b444, o445, a445, b445) -> str:
    replicate, final_pass, absolute_mean, mean_bo, mean_ba = decision(
        o444, a444, b444, o445, a445, b445
    )
    lines = [
        "# V23双随机种子复核报告",
        "",
        "## 预注册结论",
        "",
        "- GPSD配对增益复现：**{}**。".format("通过" if replicate else "未通过"),
        "- 两种子V23B Car Moderate平均为`{:.4f}`，超过V09参考值：**{}**。".format(
            absolute_mean, "是" if final_pass else "否"
        ),
        "- 判定同时要求GPSD相对公平控制和RDSA控制的独立增益，并保护三类目标。",
        "",
        "## 两种子正式KITTI 3D AP_R40",
        "",
        "| 种子/类别 | V23O | V23A | V23B | B−O | B−A |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for seed, control, rdsa, gpsd in (
        (444, o444, a444, b444),
        (445, o445, a445, b445),
    ):
        bo = PAIR.delta(gpsd, control)
        ba = PAIR.delta(gpsd, rdsa)
        for name in PAIR.CLASSES:
            lines.append(
                "| {} / {} | {} | {} | {} | {:+.4f}/{:+.4f}/{:+.4f} | "
                "{:+.4f}/{:+.4f}/{:+.4f} |".format(
                    seed,
                    name,
                    PAIR.triplet(control[name]),
                    PAIR.triplet(rdsa[name]),
                    PAIR.triplet(gpsd[name]),
                    *bo[name],
                    *ba[name],
                )
            )
    lines.extend([
        "",
        "## 两种子平均差值",
        "",
        "| 类别 | B−O E/M/H | B−A E/M/H |",
        "|---|---:|---:|",
    ])
    for name in PAIR.CLASSES:
        lines.append(
            "| {} | {:+.4f}/{:+.4f}/{:+.4f} | {:+.4f}/{:+.4f}/{:+.4f} |".format(
                name, *mean_bo[name], *mean_ba[name]
            )
        )
    lines.append("")
    return "\n".join(lines)


def self_test() -> None:
    base = {
        "Car": (68.0, 49.50, 41.50),
        "Pedestrian": (33.0, 24.5, 20.0),
        "Cyclist": (42.0, 23.0, 21.0),
    }
    a = dict(base)
    a["Car"] = (68.0, 49.57, 41.5)
    b = dict(base)
    b["Car"] = (68.0, 50.05, 41.5)
    report = build_report(base, a, b, base, a, b)
    assert "GPSD配对增益复现：**通过**" in report
    assert "超过V09参考值：**是**" in report
    print("V23双随机种子、绝对V09门槛与多类别保护自检通过")


def main() -> None:
    parser = argparse.ArgumentParser()
    names = ("o444", "a444", "b444", "o445", "a445", "b445")
    for name in names:
        parser.add_argument("--{}_log".format(name), type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    paths = [getattr(args, "{}_log".format(name)) for name in names]
    if not all(paths) or args.output is None:
        raise ValueError("six logs and output are required")
    values = [
        PAIR.parse_official_r40(
            path.read_text(encoding="utf-8", errors="replace")
        )
        for path in paths
    ]
    report = build_report(*values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print("V23双种子报告：{}".format(args.output))


if __name__ == "__main__":
    main()
