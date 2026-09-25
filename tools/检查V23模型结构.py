"""Build V23 models and audit modules, trainable scope and optimizer groups."""

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
    GroupPreservingSpatialDisparityPreAggregation,
    ResidualDisparitySpaceMicroAggregation,
)


EXPECTED_PARAMETERS = {"none": 0, "rdsa": 666, "gpsd": 25584}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model, _ = build_model(config["model"])
    pre_cfg = config["model"]["cost_preaggregation"]
    pre_type = str(pre_cfg["type"]).lower() if pre_cfg["enabled"] else "none"
    if pre_type not in EXPECTED_PARAMETERS:
        raise AssertionError("unknown V23 type: {}".format(pre_type))

    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    expected_prefixes = tuple(pre_cfg["trainable_prefixes"])
    invalid = [
        name for name, _ in trainable
        if not any(name.startswith(prefix) for prefix in expected_prefixes)
    ]
    if not trainable or invalid:
        raise AssertionError("V23 trainable scope is invalid: {}".format(invalid))

    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(
            module,
            (
                ResidualDisparitySpaceMicroAggregation,
                GroupPreservingSpatialDisparityPreAggregation,
            ),
        )
    ]
    expected_module_count = 0 if pre_type == "none" else 1
    if len(modules) != expected_module_count:
        raise AssertionError("V23 module count mismatch: {}".format(modules))
    if modules:
        expected_class = (
            ResidualDisparitySpaceMicroAggregation
            if pre_type == "rdsa"
            else GroupPreservingSpatialDisparityPreAggregation
        )
        if not isinstance(modules[0][1], expected_class):
            raise AssertionError("V23 module class does not match config type")

    added_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("depth_predictor.cost_preaggregation_s4.")
    )
    if added_parameters != EXPECTED_PARAMETERS[pre_type]:
        raise AssertionError(
            "V23 parameter mismatch: {} != {}".format(
                added_parameters, EXPECTED_PARAMETERS[pre_type]
            )
        )

    optimizer = build_optimizer(config["optimizer"], model)
    rates = sorted({float(group["lr"]) for group in optimizer.param_groups})
    base_lr = float(config["optimizer"]["lr"])
    expected_rates = [base_lr]
    if pre_type != "none":
        expected_rates.append(
            base_lr
            * float(
                config["optimizer"]["parameter_lr_multipliers"][
                    "cost_preaggregation_s4"
                ]
            )
        )
    if rates != sorted(expected_rates):
        raise AssertionError(
            "V23 optimizer rates {} != {}".format(rates, expected_rates)
        )

    print("config={}".format(args.config))
    print("model_name={}".format(config["model_name"]))
    print("cost_preaggregation_type={}".format(pre_type))
    print("modules={}".format([name for name, _ in modules]))
    print("added_parameters={}".format(added_parameters))
    print("trainable_parameters={}".format(sum(count for _, count in trainable)))
    print("trainable_tensor_count={}".format(len(trainable)))
    print("optimizer_learning_rates={}".format(rates))
    print("V23模型结构、参数量与公平训练范围检查通过")


if __name__ == "__main__":
    main()
