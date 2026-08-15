"""Audit that a V12 run changed only its pre-registered parameter scope."""

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
    group_cfg = config["model"]["groupwise_correlation"]
    allowed = tuple(group_cfg["trainable_prefixes"])
    source = normalized(model_state(args.source))
    trained = normalized(model_state(args.trained))

    missing = sorted(set(source) - set(trained))
    if missing:
        raise AssertionError("训练后checkpoint缺少原参数：{}".format(missing[:20]))
    added = sorted(set(trained) - set(source))
    if bool(group_cfg["enabled"]):
        if not added or any(
            not name.startswith("depth_predictor.groupwise_correlation_s4.")
            for name in added
        ):
            raise AssertionError("V12A新增参数越界：{}".format(added))
    elif added:
        raise AssertionError("V12O不应新增参数：{}".format(added))

    changed = []
    unchanged_allowed = []
    outside_changes = []
    for name in sorted(set(source) & set(trained)):
        same = torch.equal(source[name], trained[name])
        permitted = any(name.startswith(prefix) for prefix in allowed)
        if same and permitted:
            unchanged_allowed.append(name)
        elif not same and permitted:
            changed.append(name)
        elif not same:
            outside_changes.append(name)
    if outside_changes:
        raise AssertionError(
            "冻结区发生变化：{}".format(outside_changes[:30])
        )
    if not changed:
        raise AssertionError("允许训练的cost_agg参数没有发生任何变化")

    gate_logits = [
        value for name, value in trained.items()
        if name.startswith("depth_predictor.groupwise_correlation_s4.group_logits.")
    ]
    if bool(group_cfg["enabled"]):
        if not gate_logits:
            raise AssertionError("V12A checkpoint没有group_logits")
        for value in gate_logits:
            if not torch.isfinite(value).all():
                raise AssertionError("V12A group_logits出现NaN或Inf")
        if not any(float(value.abs().max()) > 0.0 for value in gate_logits):
            raise AssertionError("V12A group_logits训练后仍全部为零")

    print("config={}".format(args.config))
    print("source={}".format(args.source))
    print("trained={}".format(args.trained))
    print("added_tensor_count={}".format(len(added)))
    print("changed_allowed_tensor_count={}".format(len(changed)))
    print("unchanged_allowed_tensor_count={}".format(len(unchanged_allowed)))
    print("outside_changed_tensor_count=0")
    if gate_logits:
        print(
            "group_logits_abs_max={:.8g}".format(
                max(float(value.abs().max()) for value in gate_logits)
            )
        )
    print("V12训练后参数范围审计通过")


if __name__ == "__main__":
    main()
