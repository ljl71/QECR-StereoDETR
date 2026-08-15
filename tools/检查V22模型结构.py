"""Build V22 and audit the fair paired-training parameter boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config  # noqa: E402
from lib.helpers.optimizer_helper import build_optimizer  # noqa: E402
from lib.helpers.qecr_model_helper import build_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model, criterion = build_model(config["model"])
    axial = config["model"]["axial_depth_iou"]
    expected_prefixes = tuple(axial["trainable_prefixes"])
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise AssertionError("V22模型没有可训练参数")
    invalid = [
        name for name, _ in trainable
        if not any(name.startswith(prefix) for prefix in expected_prefixes)
    ]
    if invalid:
        raise AssertionError("V22训练范围越界：{}".format(invalid[:30]))
    if any(name.startswith("quality_head.") for name, _ in trainable):
        raise AssertionError("V09质量头在V22中必须冻结")
    if bool(model.axial_depth_train_only) is not True:
        raise AssertionError("V22深度分类器白名单没有生效")

    enabled = bool(axial["enabled"])
    has_loss = "axial_depth_iou" in criterion.losses
    has_weight = "loss_depth_axial_iou" in criterion.weight_dict
    if has_loss is not enabled or has_weight is not enabled:
        raise AssertionError(
            "V22损失开关不一致：enabled={} loss={} weight={}".format(
                enabled, has_loss, has_weight
            )
        )
    optimizer = build_optimizer(config["optimizer"], model)
    learning_rates = sorted({float(group["lr"]) for group in optimizer.param_groups})
    if learning_rates != [0.00002]:
        raise AssertionError("V22学习率组异常：{}".format(learning_rates))

    print("config={}".format(args.config))
    print("model_name={}".format(config["model_name"]))
    print("axial_depth_iou_enabled={}".format(enabled))
    print("total_parameters={}".format(
        sum(parameter.numel() for parameter in model.parameters())
    ))
    print("trainable_parameters={}".format(sum(count for _, count in trainable)))
    print("trainable_tensor_count={}".format(len(trainable)))
    print("optimizer_learning_rates={}".format(learning_rates))
    print("V22模型结构、损失开关和公平冻结范围检查通过")


if __name__ == "__main__":
    main()
