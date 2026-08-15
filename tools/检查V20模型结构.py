"""Build a V20 model and audit its paid-training parameter scope."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402
from lib.helpers.qecr_model_helper import build_model  # noqa: E402
from lib.helpers.optimizer_helper import build_optimizer  # noqa: E402
from lib.models.monodetr.geometry_depth_residual import (  # noqa: E402
    QueryGeometryDepthResidual,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model, _ = build_model(config["model"])
    residual_cfg = config["model"]["geometry_depth_residual"]
    expected_prefixes = tuple(residual_cfg["trainable_prefixes"])
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise AssertionError("V20模型没有可训练参数")
    invalid = [
        name for name, _ in trainable
        if not any(name.startswith(prefix) for prefix in expected_prefixes)
    ]
    if invalid:
        raise AssertionError("V20训练范围越界：{}".format(invalid))
    if not isinstance(model.depth_residual_head, QueryGeometryDepthResidual):
        raise AssertionError("没有构建V20深度残差头")
    quality_trainable = [
        name for name, _ in trainable if name.startswith("quality_head.")
    ]
    if quality_trainable:
        raise AssertionError("V09质量头在V20中必须冻结")
    optimizer = build_optimizer(config["optimizer"], model)
    learning_rates = sorted({float(group["lr"]) for group in optimizer.param_groups})
    if learning_rates != [float(config["optimizer"]["lr"])]:
        raise AssertionError("V20学习率组异常：{}".format(learning_rates))

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(count for _, count in trainable)
    head_parameters = sum(
        parameter.numel() for parameter in model.depth_residual_head.parameters()
    )
    if trainable_parameters != head_parameters:
        raise AssertionError(
            "可训练参数量{}与残差头参数量{}不一致".format(
                trainable_parameters, head_parameters
            )
        )
    print("config={}".format(args.config))
    print("model_name={}".format(config["model_name"]))
    print("use_geometry_prior={}".format(residual_cfg["use_geometry_prior"]))
    print("total_parameters={}".format(total_parameters))
    print("trainable_parameters={}".format(trainable_parameters))
    print("trainable_tensor_count={}".format(len(trainable)))
    print("optimizer_learning_rates={}".format(learning_rates))
    print("V20模型结构与冻结训练范围检查通过")


if __name__ == "__main__":
    main()

