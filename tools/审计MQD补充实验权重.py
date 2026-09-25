#!/usr/bin/env python3
"""Verify that a frozen-detector ablation changes only its declared head."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--scope", choices=("shared", "car"), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def state(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("{} has no model_state".format(path))
    return model_state


def main() -> None:
    args = parse_args()
    source = state(args.source)
    trained = state(args.trained)
    prefix = "quality_head." if args.scope == "shared" else "car_quality_head."
    expected_parameters = 66561 if args.scope == "shared" else 8257
    added = sorted(set(trained).difference(source))
    missing = sorted(set(source).difference(trained))
    changed = sorted(
        key for key in source.keys() & trained.keys()
        if not torch.equal(source[key], trained[key])
    )
    invalid_added = [key for key in added if not key.startswith(prefix)]
    if missing:
        raise RuntimeError("trained checkpoint lost tensors: {}".format(missing))
    if invalid_added:
        raise RuntimeError("unexpected added tensors: {}".format(invalid_added))
    if changed:
        raise RuntimeError("frozen tensors changed: {}".format(changed))
    buffer_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    added_buffers = [key for key in added if key.endswith(buffer_suffixes)]
    added_parameters = [key for key in added if key not in added_buffers]
    parameter_count = sum(trained[key].numel() for key in added_parameters)
    if parameter_count != expected_parameters:
        raise RuntimeError(
            "expected {} added parameters, got {}".format(
                expected_parameters, parameter_count
            )
        )
    lines = [
        "source={}".format(args.source),
        "trained={}".format(args.trained),
        "scope={}".format(args.scope),
        "added_tensor_count={}".format(len(added)),
        "added_parameter_tensor_count={}".format(len(added_parameters)),
        "added_buffer_tensor_count={}".format(len(added_buffers)),
        "added_parameter_count={}".format(parameter_count),
        "changed_existing_tensor_count=0",
        "MQD补充实验权重范围审计通过",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
