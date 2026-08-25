"""Small regression test for no-validation fixed-epoch checkpoints."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.trainer_helper import Trainer


def _build_trainer(config, model_name):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )
    logger = logging.getLogger(model_name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return Trainer(
        cfg=config,
        model=model,
        optimizer=optimizer,
        train_loader=[],
        test_loader=[],
        lr_scheduler=scheduler,
        warmup_lr_scheduler=None,
        logger=logger,
        loss=None,
        model_name=model_name,
    )


def main():
    previous_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="qecr_fixed_epoch_") as temp_dir:
        os.chdir(temp_dir)
        try:
            fresh_config = {
                "save_path": "outputs/fresh/",
                "max_epoch": 0,
                "save_frequency": 1,
                "save_all": False,
                "resume_model": False,
                "resume_if_exists": True,
                "save_final_checkpoint": True,
            }
            fresh = _build_trainer(fresh_config, "fresh_model")
            fresh.train()
            fresh_path = Path(
                "outputs/fresh/fresh_model/checkpoint_final.pth"
            )
            checkpoint = torch.load(fresh_path, map_location="cpu")
            assert checkpoint["epoch"] == 0
            assert checkpoint["model_state"]

            resume_config = dict(fresh_config)
            resume_config["save_path"] = "outputs/resume/"
            resume_config["max_epoch"] = 2
            seed = _build_trainer(resume_config, "resume_model")
            rolling_stem = Path(
                "outputs/resume/resume_model/checkpoint"
            )
            rolling_stem.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                get_checkpoint_state(
                    seed.model,
                    seed.optimizer,
                    epoch=2,
                    best_result=0,
                    best_epoch=0,
                ),
                str(rolling_stem),
            )

            resumed = _build_trainer(resume_config, "resume_model")
            assert resumed.epoch == 2
            resumed.train()
            resume_path = Path(
                "outputs/resume/resume_model/checkpoint_final.pth"
            )
            checkpoint = torch.load(resume_path, map_location="cpu")
            assert checkpoint["epoch"] == 2
            assert checkpoint["model_state"]
        finally:
            os.chdir(previous_directory)

    print("固定轮次checkpoint测试通过：无验证集保存final，滚动checkpoint可续跑")


if __name__ == "__main__":
    main()
