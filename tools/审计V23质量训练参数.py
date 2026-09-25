"""Verify that V23Q changes only the existing QLQC quality head."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def state(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    values = checkpoint["model_state"]
    return {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    args = parser.parse_args()
    source = state(args.source)
    trained = state(args.trained)
    if set(source) != set(trained):
        raise AssertionError("V23Q checkpoint tensor set changed")
    changed = [
        name for name in sorted(source)
        if not torch.equal(source[name], trained[name])
    ]
    outside = [name for name in changed if not name.startswith("quality_head.")]
    if not changed or outside:
        raise AssertionError(
            "V23Q quality-only scope violated: changed={}, outside={}".format(
                len(changed), outside[:20]
            )
        )
    print("changed_quality_tensor_count={}".format(len(changed)))
    print("outside_changed_tensor_count=0")
    print("V23Q质量头冻结训练参数审计通过")


if __name__ == "__main__":
    main()
