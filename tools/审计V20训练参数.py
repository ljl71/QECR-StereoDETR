"""Audit that V20 changes only the new depth-residual head tensors."""

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
    residual_cfg = config["model"]["geometry_depth_residual"]
    allowed = tuple(residual_cfg["trainable_prefixes"])
    source = normalized(model_state(args.source))
    trained = normalized(model_state(args.trained))

    missing = sorted(set(source) - set(trained))
    if missing:
        raise AssertionError("训练后checkpoint缺少V09参数：{}".format(missing[:20]))

    added = sorted(set(trained) - set(source))
    if not added or any(
        not any(name.startswith(prefix) for prefix in allowed)
        for name in added
    ):
        raise AssertionError("V20新增参数越界：{}".format(added))

    outside_changes = []
    for name in sorted(set(source) & set(trained)):
        if not torch.equal(source[name], trained[name]):
            outside_changes.append(name)
    if outside_changes:
        raise AssertionError(
            "V09冻结区发生变化：{}".format(outside_changes[:30])
        )

    added_values = [trained[name] for name in added]
    if any(not torch.isfinite(value).all() for value in added_values):
        raise AssertionError("V20残差头出现NaN或Inf")
    if not any(float(value.abs().max()) > 0.0 for value in added_values):
        raise AssertionError("V20残差头训练后仍全部为零")

    print("config={}".format(args.config))
    print("source={}".format(args.source))
    print("trained={}".format(args.trained))
    print("added_tensor_count={}".format(len(added)))
    print("added_parameter_count={}".format(
        sum(value.numel() for value in added_values)
    ))
    print("residual_abs_max={:.8g}".format(
        max(float(value.abs().max()) for value in added_values)
    ))
    print("outside_changed_tensor_count=0")
    print("V20训练后参数范围审计通过")


if __name__ == "__main__":
    main()
