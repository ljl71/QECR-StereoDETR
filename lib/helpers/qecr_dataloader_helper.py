"""Dataloader builder for the calibration-aware QECR KITTI dataset."""

import numpy as np
from torch.utils.data import DataLoader

from lib.datasets.kitti.qecr_kitti_dataset import QECRKITTIDataset


def _worker_init(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_qecr_dataloader(cfg, workers=8):
    if cfg["type"] != "KITTI":
        raise NotImplementedError(
            "{} is not supported by QECR".format(cfg["type"])
        )
    train_set = QECRKITTIDataset(split=cfg["train_split"], cfg=cfg)
    test_set = QECRKITTIDataset(split=cfg["test_split"], cfg=cfg)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg["batch_size"],
        num_workers=workers,
        worker_init_fn=_worker_init,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg["batch_size"],
        num_workers=workers,
        worker_init_fn=_worker_init,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, test_loader

