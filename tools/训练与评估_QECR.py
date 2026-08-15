"""AutoDL entry point for non-distillation QECR-StereoDETR."""

import argparse
import datetime
import os
import sys
import warnings

import torch

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.helpers.config_helper import load_config  # noqa: E402
from lib.helpers.qecr_dataloader_helper import build_qecr_dataloader  # noqa: E402
from lib.helpers.qecr_model_helper import build_model  # noqa: E402
from lib.helpers.optimizer_helper import build_optimizer  # noqa: E402
from lib.helpers.qecr_trainer_helper import QECRTrainer  # noqa: E402
from lib.helpers.scheduler_helper import build_lr_scheduler  # noqa: E402
from lib.helpers.tester_helper import Tester  # noqa: E402
from lib.helpers.utils_helper import create_logger, set_random_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate non-distillation QECR-StereoDETR"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML experiment configuration",
    )
    parser.add_argument(
        "-e",
        "--evaluate_only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--quality_score_power",
        type=float,
        default=None,
        help=(
            "evaluation-time exponent for the learned 3D quality factor; "
            "0 disables it and 1 reproduces the trained V06 score"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(args.config)
    cfg = load_config(args.config)
    if args.quality_score_power is not None:
        if args.quality_score_power < 0.0:
            raise ValueError("--quality_score_power must be non-negative")
        quality_cfg = cfg.setdefault("model", {}).setdefault(
            "quality_ranking", {}
        )
        if not quality_cfg.get("enabled", False):
            raise ValueError(
                "--quality_score_power requires quality_ranking.enabled"
            )
        quality_cfg["score_power"] = float(args.quality_score_power)
    set_random_seed(cfg.get("random_seed", 444))

    model_name = cfg["model_name"]
    output_path = os.path.join(
        "./" + cfg["trainer"]["save_path"],
        model_name,
    )
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(
        output_path,
        "train.log.{}".format(
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
    )
    logger = create_logger(log_file)
    if args.quality_score_power is not None:
        logger.info(
            "Quality score power override: %.4f",
            args.quality_score_power,
        )

    workers = int(cfg["dataset"].get("workers", 8))
    train_loader, test_loader = build_qecr_dataloader(
        cfg["dataset"],
        workers=workers,
    )
    model, loss = build_model(cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg["trainer"]["gpu_ids"].split(",")))
    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    if args.evaluate_only:
        tester = Tester(
            cfg=cfg["tester"],
            model=model,
            dataloader=test_loader,
            logger=logger,
            train_cfg=cfg["trainer"],
            model_name=model_name,
        )
        tester.test()
        return

    optimizer = build_optimizer(cfg["optimizer"], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(
        cfg["lr_scheduler"],
        optimizer,
        last_epoch=-1,
    )
    trainer = QECRTrainer(
        cfg=cfg["trainer"],
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        lr_scheduler=lr_scheduler,
        warmup_lr_scheduler=warmup_lr_scheduler,
        logger=logger,
        loss=loss,
        model_name=model_name,
    )
    tester = Tester(
        cfg=cfg["tester"],
        model=trainer.model,
        dataloader=test_loader,
        logger=logger,
        train_cfg=cfg["trainer"],
        model_name=model_name,
    )
    if cfg["dataset"]["test_split"] != "test":
        trainer.tester = tester

    logger.info("Training non-distillation QECR-StereoDETR")
    logger.info("Batch Size: %d", cfg["dataset"]["batch_size"])
    logger.info("Learning Rate: %f", cfg["optimizer"]["lr"])
    trainer.train()

    if cfg["dataset"]["test_split"] != "test":
        tester.test()


if __name__ == "__main__":
    main()
