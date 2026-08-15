"""Verify that V22 changes only the shared depth-classifier whitelist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402


def model_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if "model_state" not in checkpoint:
        raise KeyError("checkpoint has no model_state: {}".format(path))
    return {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in checkpoint["model_state"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    allowed = tuple(
        config["model"]["axial_depth_iou"]["trainable_prefixes"]
    )
    source = model_state(args.source)
    trained = model_state(args.trained)
    missing = sorted(set(source) - set(trained))
    added = sorted(set(trained) - set(source))
    if missing or added:
        raise AssertionError(
            "V22不得增删推理参数：missing={} added={}".format(
                missing[:20], added[:20]
            )
        )

    changed = []
    outside = []
    for name in sorted(source):
        if not torch.equal(source[name], trained[name]):
            changed.append(name)
            if not any(name.startswith(prefix) for prefix in allowed):
                outside.append(name)
        if not torch.isfinite(trained[name]).all():
            raise AssertionError("V22参数出现NaN或Inf：{}".format(name))
    if outside:
        raise AssertionError("V22冻结区发生变化：{}".format(outside[:30]))
    if not changed:
        raise AssertionError("V22训练后没有任何白名单参数发生变化")

    print("config={}".format(args.config))
    print("source={}".format(args.source))
    print("trained={}".format(args.trained))
    print("added_tensor_count=0")
    print("removed_tensor_count=0")
    print("changed_tensor_count={}".format(len(changed)))
    print("changed_parameter_count={}".format(
        sum(trained[name].numel() for name in changed)
    ))
    print("outside_changed_tensor_count=0")
    print("V22训练后参数范围与零推理参数增量审计通过")


if __name__ == "__main__":
    main()
