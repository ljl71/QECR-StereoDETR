"""Build V12 models and audit structure, trainable scope and optimizer groups."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402
from lib.helpers.optimizer_helper import build_optimizer  # noqa: E402
from lib.helpers.qecr_model_helper import build_model  # noqa: E402
from lib.models.monodetr.depth_predictor.depth_predictor_lightstereo import (  # noqa: E402
    BaselinePreservingGroupwiseCorrelation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model, _ = build_model(config["model"])
    group_cfg = config["model"]["groupwise_correlation"]
    expected_prefixes = tuple(group_cfg["trainable_prefixes"])
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise AssertionError("V12模型没有可训练参数")
    invalid = [
        name for name, _ in trainable
        if not any(name.startswith(prefix) for prefix in expected_prefixes)
    ]
    if invalid:
        raise AssertionError("V12训练范围越界：{}".format(invalid))

    modules = [
        name for name, module in model.named_modules()
        if isinstance(module, BaselinePreservingGroupwiseCorrelation)
    ]
    expected_count = 1 if bool(group_cfg["enabled"]) else 0
    if len(modules) != expected_count:
        raise AssertionError(
            "V12门控数量错误：实际{}，预期{}".format(modules, expected_count)
        )
    gate_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "groupwise_correlation_s4" in name
    )
    if gate_parameters > 1000:
        raise AssertionError("V12门控参数超过预注册上限：{}".format(gate_parameters))

    optimizer = build_optimizer(config["optimizer"], model)
    learning_rates = sorted({float(group["lr"]) for group in optimizer.param_groups})
    expected_rates = [float(config["optimizer"]["lr"])]
    if expected_count:
        expected_rates.append(
            float(config["optimizer"]["lr"])
            * float(
                config["optimizer"]["parameter_lr_multipliers"][
                    "groupwise_correlation_s4"
                ]
            )
        )
    expected_rates = sorted(set(expected_rates))
    if learning_rates != expected_rates:
        raise AssertionError(
            "V12实际学习率组错误：{}，预期{}".format(
                learning_rates, expected_rates
            )
        )

    print("config={}".format(args.config))
    print("model_name={}".format(config["model_name"]))
    print("groupwise_enabled={}".format(bool(group_cfg["enabled"])))
    print("groupwise_scale={}".format(int(group_cfg["scale"])))
    print("groupwise_groups={}".format(int(group_cfg["num_groups"])))
    print("groupwise_modules={}".format(modules))
    print("gate_parameters={}".format(gate_parameters))
    print("trainable_parameters={}".format(sum(count for _, count in trainable)))
    print("trainable_tensor_count={}".format(len(trainable)))
    print("optimizer_learning_rates={}".format(learning_rates))
    print("V12模型结构与公平训练范围检查通过")


if __name__ == "__main__":
    main()
