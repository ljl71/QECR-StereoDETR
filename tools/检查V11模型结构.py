"""Build a V11 model and audit its trainable scope before paid training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402
from lib.helpers.qecr_model_helper import build_model  # noqa: E402
from lib.helpers.optimizer_helper import build_optimizer  # noqa: E402
from lib.models.monodetr.depth_predictor.depth_predictor_lightstereo import (  # noqa: E402
    BaselinePreservingDynamicUpsample,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model, _ = build_model(config["model"])
    dynamic_cfg = config["model"]["dynamic_depth_upsampling"]
    expected_prefixes = tuple(dynamic_cfg["trainable_prefixes"])
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise AssertionError("V11模型没有可训练参数")
    invalid = [
        name for name, _ in trainable
        if not any(name.startswith(prefix) for prefix in expected_prefixes)
    ]
    if invalid:
        raise AssertionError("V11训练范围越界：{}".format(invalid))

    dynamic_modules = [
        name for name, module in model.named_modules()
        if isinstance(module, BaselinePreservingDynamicUpsample)
    ]
    expected_dynamic_count = len(dynamic_cfg["stages"])
    if len(dynamic_modules) != expected_dynamic_count:
        raise AssertionError(
            "动态上采样数量错误：实际{}，预期{}".format(
                dynamic_modules, expected_dynamic_count
            )
        )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(count for _, count in trainable)
    dynamic_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "offset_predictor" in name
    )
    quality_trainable = [
        name for name, _ in trainable if name.startswith("quality_head.")
    ]
    if quality_trainable:
        raise AssertionError("V09质量头在V11中必须冻结")
    optimizer = build_optimizer(config["optimizer"], model)
    learning_rates = sorted(
        {float(group["lr"]) for group in optimizer.param_groups}
    )
    expected_learning_rates = [float(config["optimizer"]["lr"])]
    if expected_dynamic_count:
        expected_learning_rates.append(
            float(config["optimizer"]["lr"])
            * float(
                config["optimizer"]["parameter_lr_multipliers"][
                    "offset_predictor"
                ]
            )
        )
    expected_learning_rates = sorted(set(expected_learning_rates))
    if learning_rates != expected_learning_rates:
        raise AssertionError(
            "V11实际学习率组错误：{}，预期{}".format(
                learning_rates, expected_learning_rates
            )
        )

    print("config={}".format(args.config))
    print("model_name={}".format(config["model_name"]))
    print("dynamic_stages={}".format(list(dynamic_cfg["stages"])))
    print("dynamic_modules={}".format(dynamic_modules))
    print("total_parameters={}".format(total_parameters))
    print("trainable_parameters={}".format(trainable_parameters))
    print("dynamic_parameters={}".format(dynamic_parameters))
    print("optimizer_learning_rates={}".format(learning_rates))
    print("trainable_tensor_count={}".format(len(trainable)))
    print("V11模型结构与训练范围检查通过")


if __name__ == "__main__":
    main()
