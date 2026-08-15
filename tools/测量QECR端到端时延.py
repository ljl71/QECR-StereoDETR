"""Measure synchronized batch-one model-forward latency for a checkpoint."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from QECR运行时引导 import install_runtime_module  # noqa: E402

install_runtime_module()

from lib.helpers.config_helper import load_config  # noqa: E402
from lib.helpers.qecr_dataloader_helper import build_qecr_dataloader  # noqa: E402
from lib.helpers.qecr_model_helper import build_model  # noqa: E402
from lib.helpers.save_helper import load_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--quality_score_power", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for latency measurement")
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
    cfg["dataset"]["batch_size"] = 1
    _, loader = build_qecr_dataloader(
        cfg["dataset"],
        workers=int(cfg["dataset"].get("workers", 8)),
    )
    model, _ = build_model(cfg["model"])
    device = torch.device("cuda:0")
    model.to(device).eval()
    load_checkpoint(
        model=model,
        optimizer=None,
        filename=args.checkpoint,
        map_location=device,
        logger=None,
    )

    timings = []
    total_needed = args.warmup + args.steps
    with torch.no_grad():
        for index, (inputs, calibs, targets, info) in enumerate(loader):
            inputs = inputs.to(device)
            calibs = calibs.to(device)
            img_sizes = info["img_size_croped"].to(device)
            img_sizes_ori = info["img_size_original"].to(device)
            img_sizes_upper = info["upper"].to(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(
                inputs,
                calibs,
                targets,
                img_sizes,
                img_sizes_ori,
                img_sizes_upper,
                dn_args=0,
            )
            end.record()
            torch.cuda.synchronize()
            if index >= args.warmup:
                timings.append(float(start.elapsed_time(end)))
            if index + 1 >= total_needed:
                break

    if len(timings) < args.steps:
        raise RuntimeError(
            "validation set provided only {} timed steps".format(len(timings))
        )
    timings.sort()
    p90_index = min(
        len(timings) - 1,
        int(round(0.90 * (len(timings) - 1))),
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print("GPU:", torch.cuda.get_device_name(0))
    print("quality_score_power:", args.quality_score_power)
    print("parameters:", parameters)
    print("median_ms:", statistics.median(timings))
    print("mean_ms:", statistics.mean(timings))
    print("p90_ms:", timings[p90_index])
    print("fps_from_median:", 1000.0 / statistics.median(timings))


if __name__ == "__main__":
    main()
