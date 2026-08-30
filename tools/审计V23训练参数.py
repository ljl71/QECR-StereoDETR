"""Audit that a V23 run changed only the pre-registered parameter scope."""

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
    return checkpoint["model_state"]


def normalized(state):
    return {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in state.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    pre_cfg = config["model"]["cost_preaggregation"]
    pre_type = str(pre_cfg["type"]).lower() if pre_cfg["enabled"] else "none"
    allowed = tuple(pre_cfg["trainable_prefixes"])
    source = normalized(model_state(args.source))
    trained = normalized(model_state(args.trained))

    missing = sorted(set(source) - set(trained))
    if missing:
        raise AssertionError("trained checkpoint lost tensors: {}".format(missing[:20]))
    added = sorted(set(trained) - set(source))
    added_prefix = "depth_predictor.cost_preaggregation_s4."
    if pre_type == "none":
        if added:
            raise AssertionError("V23O must not add tensors: {}".format(added))
    elif not added or any(not name.startswith(added_prefix) for name in added):
        raise AssertionError("V23 candidate added tensors out of scope: {}".format(added))

    changed = []
    outside = []
    for name in sorted(set(source) & set(trained)):
        if torch.equal(source[name], trained[name]):
            continue
        if any(name.startswith(prefix) for prefix in allowed):
            changed.append(name)
        else:
            outside.append(name)
    if outside:
        raise AssertionError("frozen tensors changed: {}".format(outside[:30]))
    if not any(name.startswith("depth_predictor.cost_agg.") for name in changed):
        raise AssertionError("existing cost_agg tensors did not change")

    added_values = [trained[name] for name in added]
    if added_values:
        if any(not torch.isfinite(value).all() for value in added_values):
            raise AssertionError("V23 added tensors contain NaN or Inf")
        if not any(float(value.abs().max()) > 0.0 for value in added_values):
            raise AssertionError("V23 added tensors remained all zero")

    print("config={}".format(args.config))
    print("source={}".format(args.source))
    print("trained={}".format(args.trained))
    print("cost_preaggregation_type={}".format(pre_type))
    print("added_tensor_count={}".format(len(added)))
    print("changed_existing_tensor_count={}".format(len(changed)))
    print("outside_changed_tensor_count=0")
    if added_values:
        print(
            "added_abs_max={:.8g}".format(
                max(float(value.abs().max()) for value in added_values)
            )
        )
    print("V23训练后参数范围审计通过")


if __name__ == "__main__":
    main()
