"""Test strict=False checkpoint loading with an explicit missing-key whitelist."""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.save_helper import load_checkpoint  # noqa: E402


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector = nn.Linear(4, 4)
        self.quality_head = nn.Linear(4, 1)


def _save(path: Path, state) -> None:
    torch.save(
        {
            "epoch": 1,
            "model_state": state,
            "optimizer_state": None,
            "best_result": 0.0,
            "best_epoch": 1,
        },
        path,
    )


def _expect_runtime_error(callback) -> None:
    try:
        callback()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def main() -> None:
    logger = logging.getLogger("checkpoint_whitelist_test")
    model = _ToyModel()
    complete = model.state_dict()
    detector_only = {
        key: value for key, value in complete.items()
        if key.startswith("detector.")
    }

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        valid_path = root / "valid.pth"
        _save(valid_path, detector_only)
        load_checkpoint(
            model=_ToyModel(),
            optimizer=None,
            filename=str(valid_path),
            map_location="cpu",
            logger=logger,
            strict=False,
            allowed_missing_prefixes=["quality_head."],
            allow_unexpected=False,
        )

        missing_detector_path = root / "missing_detector.pth"
        _save(missing_detector_path, {})
        _expect_runtime_error(
            lambda: load_checkpoint(
                model=_ToyModel(),
                optimizer=None,
                filename=str(missing_detector_path),
                map_location="cpu",
                logger=logger,
                strict=False,
                allowed_missing_prefixes=["quality_head."],
                allow_unexpected=False,
            )
        )

        unexpected_path = root / "unexpected.pth"
        unexpected_state = dict(detector_only)
        unexpected_state["unknown.weight"] = torch.zeros(1)
        _save(unexpected_path, unexpected_state)
        _expect_runtime_error(
            lambda: load_checkpoint(
                model=_ToyModel(),
                optimizer=None,
                filename=str(unexpected_path),
                map_location="cpu",
                logger=logger,
                strict=False,
                allowed_missing_prefixes=["quality_head."],
                allow_unexpected=False,
            )
        )

    print("checkpoint兼容白名单测试通过：只允许quality_head.*缺失")


if __name__ == "__main__":
    main()
