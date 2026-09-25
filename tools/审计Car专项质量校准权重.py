#!/usr/bin/env python3
"""Verify that Car residual training leaves every pretrained tensor intact."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_state(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["model_state"]


def main() -> None:
    args = parse_args()
    source = load_state(args.source)
    trained = load_state(args.trained)
    added = sorted(set(trained).difference(source))
    missing = sorted(set(source).difference(trained))
    changed_existing = sorted(
        key
        for key in source.keys() & trained.keys()
        if not torch.equal(source[key], trained[key])
    )
    invalid_added = [
        key for key in added if not key.startswith("car_quality_head.")
    ]
    if missing:
        raise RuntimeError("trained checkpoint lost tensors: {}".format(missing))
    if invalid_added:
        raise RuntimeError(
            "unexpected added tensors: {}".format(invalid_added)
        )
    if changed_existing:
        raise RuntimeError(
            "pretrained tensors changed: {}".format(changed_existing)
        )
    added_parameter_count = sum(trained[key].numel() for key in added)
    if added_parameter_count != 8257:
        raise RuntimeError(
            "expected 8257 added parameters, got {}".format(
                added_parameter_count
            )
        )
    max_abs = max(
        (float(trained[key].abs().max()) for key in added),
        default=0.0,
    )
    lines = [
        "source={}".format(args.source),
        "trained={}".format(args.trained),
        "added_tensor_count={}".format(len(added)),
        "added_parameter_count={}".format(added_parameter_count),
        "changed_existing_tensor_count=0",
        "car_head_abs_max={:.8g}".format(max_abs),
        "Car专项质量校准训练后参数审计通过",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
