"""Compare the accepted GPSD checkpoint with three-epoch QLQC recalibration."""

from __future__ import annotations

import argparse
import importlib.util
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


def decision(gpsd, quality):
    change = PAIR.delta(quality, gpsd)
    protected = (
        change["Car"][0] >= -0.10
        and change["Car"][2] >= -0.10
        and change["Pedestrian"][1] >= -0.10
        and change["Pedestrian"][2] >= -0.10
        and change["Cyclist"][1] >= -0.10
        and change["Cyclist"][2] >= -0.10
    )
    use_quality = (
        quality["Car"][1] >= gpsd["Car"][1]
        and quality["Car"][1] >= PAIR.V09_CAR_MODERATE
        and protected
    )
    return use_quality, change


def build_report(gpsd, quality, gpsd_log, quality_log) -> str:
    use_quality, change = decision(gpsd, quality)
    winner = "V23Q" if use_quality else "V23B"
    lines = [
        "# V23最终质量重标定结果",
        "",
        "- 自动选择的最终验证权重：**{}**。".format(winner),
        "- 只有V23Q不降低Car Moderate、超过V09参考值并满足多类别保护时，才替换V23B。",
        "",
        "| 类别 | V23B E/M/H | V23Q E/M/H | Q−B E/M/H |",
        "|---|---:|---:|---:|",
    ]
    for name in PAIR.CLASSES:
        lines.append(
            "| {} | {} | {} | {:+.4f}/{:+.4f}/{:+.4f} |".format(
                name,
                PAIR.triplet(gpsd[name]),
                PAIR.triplet(quality[name]),
                *change[name],
            )
        )
    lines.extend([
        "",
        "- V23B日志：`{}`".format(gpsd_log),
        "- V23Q日志：`{}`".format(quality_log),
        "",
    ])
    return "\n".join(lines)


def self_test() -> None:
    base = {
        "Car": (68.0, 49.9, 41.5),
        "Pedestrian": (33.0, 24.5, 20.0),
        "Cyclist": (42.0, 23.0, 21.0),
    }
    quality = dict(base)
    quality["Car"] = (68.0, 50.0, 41.5)
    assert decision(base, quality)[0]
    quality["Car"] = (68.0, 49.8, 41.5)
    assert not decision(base, quality)[0]
    print("V23质量重标定选择与多类别保护规则自检通过")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpsd_log", type=Path)
    parser.add_argument("--quality_log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.gpsd_log, args.quality_log, args.output)):
        raise ValueError("gpsd_log, quality_log and output are required")
    gpsd = PAIR.parse_official_r40(
        args.gpsd_log.read_text(encoding="utf-8", errors="replace")
    )
    quality = PAIR.parse_official_r40(
        args.quality_log.read_text(encoding="utf-8", errors="replace")
    )
    report = build_report(gpsd, quality, args.gpsd_log, args.quality_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print("V23最终选择报告：{}".format(args.output))


if __name__ == "__main__":
    main()
