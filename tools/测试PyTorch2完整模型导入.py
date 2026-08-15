#!/usr/bin/env python3
"""Verify the real StereoDETR import chain under the pinned PyTorch 2.0 stack."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from lib.helpers.qecr_model_helper import build_model  # noqa: F401
from lib.models.monodetr.ops.modules import (
    MSDeformAttn,  # noqa: F401
    MSDeformAttn_cross,  # noqa: F401
    MultiheadAttention,
)


def main() -> None:
    attention = MultiheadAttention(256, 8)
    assert attention.out_proj.weight.shape == (256, 256)
    print("PyTorch:", torch.__version__)
    print("out_proj:", type(attention.out_proj).__name__)
    print("StereoDETR PyTorch 2.0 完整模型导入链测试通过")


if __name__ == "__main__":
    main()
