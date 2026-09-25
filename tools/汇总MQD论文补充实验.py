#!/usr/bin/env python3
"""Build a factual AP_R40 table from MQD supplementary experiment logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


CLASSES = ("Car", "Pedestrian", "Cyclist")
HEADERS = {
    "Car": "Car AP_R40@0.70, 0.70, 0.70:",
    "Pedestrian": "Pedestrian AP_R40@0.50, 0.50, 0.50:",
    "Cyclist": "Cyclist AP_R40@0.50, 0.50, 0.50:",
}
AP_PATTERN = re.compile(
    r"3d\s+AP:\s*([-+0-9.]+),\s*([-+0-9.]+),\s*([-+0-9.]+)"
)


def parse_result(text: str) -> Dict[str, Tuple[float, float, float]]:
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
            if active is not None:
                match = AP_PATTERN.search(line)
                if match:
                    history[active].append(
                        tuple(float(value) for value in match.groups())
                    )
                    active = None
    missing = [name for name, values in history.items() if not values]
    if missing:
        raise ValueError("log misses R40 results: {}".format(", ".join(missing)))
    return {name: history[name][-1] for name in CLASSES}


def parse_entry(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("entry must be name=path")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def triplet(values: Tuple[float, float, float]) -> str:
    return "{:.4f} / {:.4f} / {:.4f}".format(*values)


def delta(
    values: Tuple[float, float, float],
    reference: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    return tuple(value - base for value, base in zip(values, reference))


def append_controlled_table(
    lines: List[str],
    title: str,
    entries: Sequence[Tuple[str, Dict[str, Tuple[float, float, float]]]],
    reference_name: str,
) -> None:
    lookup = dict(entries)
    if reference_name not in lookup:
        return
    reference = lookup[reference_name]
    lines.extend([
        "", "## {}".format(title), "",
        "| 版本 | ΔCar E/M/H | ΔPedestrian E/M/H | ΔCyclist E/M/H |",
        "|---|---:|---:|---:|",
    ])
    for name, values in entries:
        if name == reference_name:
            continue
        lines.append("| {} | {} | {} | {} |".format(
            name,
            triplet(delta(values["Car"], reference["Car"])),
            triplet(delta(values["Pedestrian"], reference["Pedestrian"])),
            triplet(delta(values["Cyclist"], reference["Cyclist"])),
        ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parsed = []
    for spec in args.entry:
        name, path = parse_entry(spec)
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed.append((name, path, parse_result(
            path.read_text(encoding="utf-8", errors="replace")
        )))

    lines = [
        "# MQD论文补充实验汇总",
        "",
        "本表只转录各日志最后一组正式KITTI 3D AP_R40，不自动放宽或追加判定门槛。",
        "",
        "| 版本 | Car E/M/H | Pedestrian E/M/H | Cyclist E/M/H |",
        "|---|---:|---:|---:|",
    ]
    for name, _, values in parsed:
        lines.append("| {} | {} | {} | {} |".format(
            name, triplet(values["Car"]), triplet(values["Pedestrian"]),
            triplet(values["Cyclist"]),
        ))
    values_only = [(name, values) for name, _, values in parsed]
    shared_entries = [
        item for item in values_only
        if item[0].startswith("S0-") or item[0].startswith("S1")
        or item[0].startswith("S2") or item[0].startswith("S3")
    ]
    boundary_entries = [
        item for item in values_only
        if item[0].startswith("B0-") or item[0].startswith("B1")
        or item[0].startswith("B2") or item[0].startswith("B3")
        or item[0].startswith("B4")
    ]
    append_controlled_table(
        lines, "CAGS相对同一起点的变化", shared_entries,
        "S0-StereoDETR",
    )
    append_controlled_table(
        lines, "BCDF相对共享MQD的变化", boundary_entries,
        "B0-Shared-MQD",
    )
    lines.extend(["", "## 审计路径", ""])
    for name, path, _ in parsed:
        lines.append("- {}：`{}`".format(name, path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
