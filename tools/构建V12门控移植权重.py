"""Build a V09 checkpoint containing only a trained V12 gate transplant."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch


GATE_PREFIX = "depth_predictor.groupwise_correlation_s4."


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if "model_state" not in checkpoint:
        raise KeyError("checkpoint has no model_state: {}".format(path))
    return checkpoint


def normalized_items(state):
    return {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in state.items()
    }


def build(base_path: Path, output: Path, donor_path: Optional[Path]):
    base = load_checkpoint(base_path)
    base_state = normalized_items(base["model_state"])
    if any(name.startswith(GATE_PREFIX) for name in base_state):
        raise AssertionError("V09 base unexpectedly contains a V12 gate")

    output_state = OrderedDict(
        (name, value.clone()) for name, value in base_state.items()
    )
    transplanted = []
    if donor_path is not None:
        donor = load_checkpoint(donor_path)
        donor_state = normalized_items(donor["model_state"])
        transplanted = sorted(
            name for name in donor_state if name.startswith(GATE_PREFIX)
        )
        expected = [
            GATE_PREFIX + "context.weight",
            GATE_PREFIX + "group_logits.bias",
            GATE_PREFIX + "group_logits.weight",
        ]
        if transplanted != expected:
            raise AssertionError(
                "unexpected donor gate tensors: {}".format(transplanted)
            )
        for name in transplanted:
            value = donor_state[name]
            if not torch.isfinite(value).all():
                raise AssertionError("non-finite donor gate: {}".format(name))
            output_state[name] = value.clone()

    # Prove every shared tensor is bitwise identical to the V09 source.
    for name, value in base_state.items():
        if not torch.equal(value, output_state[name]):
            raise AssertionError("V09 tensor changed during transplant: {}".format(name))

    result = dict(base)
    result["model_state"] = output_state
    result["optimizer_state"] = None
    result["v12_gate_transplant"] = {
        "base": str(base_path),
        "donor": None if donor_path is None else str(donor_path),
        "transplanted_tensors": transplanted,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print("base_shared_tensor_count={}".format(len(base_state)))
    print("transplanted_tensor_count={}".format(len(transplanted)))
    print("all_base_tensors_bitwise_identical=True")
    print("output={}".format(output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.base, args.output, args.donor)


if __name__ == "__main__":
    main()
