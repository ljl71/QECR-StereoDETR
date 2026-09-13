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
from lib.helpers.parallel_helper import build_parallel_model  # noqa: E402
from lib.helpers.parallel_helper import parse_gpu_ids  # noqa: E402
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="override dataset.batch_size; this is the global batch size",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="override trainer.gpu_ids, for example 0 or 0,1",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp",
        dest="amp_enabled",
        action="store_true",
        help="enable CUDA float16 automatic mixed precision for training",
    )
    amp_group.add_argument(
        "--no_amp",
        dest="amp_enabled",
        action="store_false",
        help="disable automatic mixed precision",
    )
    parser.set_defaults(amp_enabled=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(args.config)
    cfg = load_config(args.config)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch_size must be positive")
        cfg["dataset"]["batch_size"] = int(args.batch_size)
    if args.gpu_ids is not None:
        cfg["trainer"]["gpu_ids"] = args.gpu_ids
    if args.amp_enabled is not None:
        cfg["trainer"]["amp_enabled"] = bool(args.amp_enabled)
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

    gpu_ids = parse_gpu_ids(cfg["trainer"]["gpu_ids"])
    if not torch.cuda.is_available():
        raise RuntimeError("StereoDETR training/evaluation requires CUDA")
    visible_device_count = torch.cuda.device_count()
    invalid_gpu_ids = [
        device_id for device_id in gpu_ids
        if device_id >= visible_device_count
    ]
    if invalid_gpu_ids:
        raise ValueError(
            "trainer.gpu_ids={} exceeds the {} CUDA devices visible to "
            "this process".format(invalid_gpu_ids, visible_device_count)
        )
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device("cuda", gpu_ids[0])

    workers = int(cfg["dataset"].get("workers", 8))
    train_loader, test_loader = build_qecr_dataloader(
        cfg["dataset"],
        workers=workers,
    )
    model, loss = build_model(cfg["model"])
    model = build_parallel_model(model, gpu_ids)

    global_batch_size = int(cfg["dataset"]["batch_size"])
    full_batch_chunks = [
        global_batch_size // len(gpu_ids)
        + (1 if index < global_batch_size % len(gpu_ids) else 0)
        for index in range(len(gpu_ids))
    ]
    logger.info("CUDA devices: %s", gpu_ids)
    logger.info("GPU names: %s", [
        torch.cuda.get_device_name(device_id) for device_id in gpu_ids
    ])
    logger.info("Global batch size: %d", global_batch_size)
    logger.info("Nominal per-device chunks: %s", full_batch_chunks)
    logger.info(
        "AMP training: %s",
        bool(cfg["trainer"].get("amp_enabled", False)),
    )

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
    logger.info("Batch Size (global): %d", cfg["dataset"]["batch_size"])
    logger.info("Learning Rate: %f", cfg["optimizer"]["lr"])
    trainer.train()

    if cfg["dataset"]["test_split"] != "test":
        tester.test()


if __name__ == "__main__":
    main()
