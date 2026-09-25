#!/usr/bin/env python3
"""Compare fixed-query decision evidence with camera-space 3D IoU.

Each ``--variant`` is ``name::config::checkpoint``.  The script runs every
variant on the same KITTI validation loader, exports query-level records, and
reports correlation, score-decile reliability, threshold purity and Top-K
coverage.  It does not alter checkpoints or detector predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
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
from lib.models.monodetr.quality_ranking import (  # noqa: E402
    build_3d_iou_quality_targets,
)


CLASS_NAMES = ("Car", "Pedestrian", "Cyclist")
CLASS_THRESHOLDS = (0.7, 0.5, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="name::config::checkpoint; repeat for every model",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=0)
    return parser.parse_args()


def parse_variant(spec: str) -> Tuple[str, Path, Path]:
    pieces = spec.split("::", 2)
    if len(pieces) != 3 or not all(piece.strip() for piece in pieces):
        raise ValueError("variant must be name::config::checkpoint")
    name, config, checkpoint = pieces
    return name.strip(), Path(config), Path(checkpoint)


def prepare_targets(
    targets: Mapping[str, torch.Tensor],
    batch_size: int,
) -> List[Dict[str, torch.Tensor]]:
    filtered_keys = {
        "labels", "boxes", "calibs", "depth", "size_3d", "src_size_3d",
        "heading_bin", "heading_res", "boxes_ry", "boxes_3d",
        "sample_points",
    }
    image_keys = {
        "disp", "disp_candidates", "disp_candidate_weights",
        "random_flip_flag", "random_crop_flag", "random_mix_flag",
        "random_switch_flag", "crop_scale", "stereo_fb", "stereo_direction",
    }
    mask = targets["mask_2d"]
    result: List[Dict[str, torch.Tensor]] = []
    for batch_index in range(batch_size):
        item: Dict[str, torch.Tensor] = {}
        for key, value in targets.items():
            if key in filtered_keys:
                item[key] = value[batch_index][mask[batch_index]]
            elif key in image_keys:
                item[key] = value[batch_index]
        result.append(item)
    return result


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x: Iterable[float], y: Iterable[float]) -> Dict[str, object]:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[finite]
    y_array = y_array[finite]
    if len(x_array) < 2 or np.std(x_array) == 0.0 or np.std(y_array) == 0.0:
        return {"count": int(len(x_array)), "pearson": None, "spearman": None}
    return {
        "count": int(len(x_array)),
        "pearson": float(np.corrcoef(x_array, y_array)[0, 1]),
        "spearman": float(
            np.corrcoef(rankdata(x_array), rankdata(y_array))[0, 1]
        ),
    }


def decile_summary(records: Sequence[Mapping[str, object]]) -> List[Dict[str, float]]:
    if not records:
        return []
    scores = np.asarray([row["decision_score"] for row in records], dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    groups = np.array_split(order, 10)
    result = []
    for index, group in enumerate(groups, start=1):
        if len(group) == 0:
            continue
        result.append({
            "decile": index,
            "count": int(len(group)),
            "mean_score": float(np.mean(scores[group])),
            "mean_iou": float(np.mean([
                records[item]["iou_target"] for item in group
            ])),
        })
    return result


def summarize(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    variants = sorted({str(row["variant"]) for row in records})
    for variant in variants:
        variant_rows = [row for row in records if row["variant"] == variant]
        variant_result: Dict[str, object] = {}
        for class_index, class_name in enumerate(CLASS_NAMES):
            rows = [row for row in variant_rows if row["class_index"] == class_index]
            positive = [row for row in rows if row["iou_target"] > 0.0]
            retained = [row for row in rows if row["retained"]]
            threshold = CLASS_THRESHOLDS[class_index]
            qualifying = [row for row in rows if row["iou_target"] >= threshold]
            retained_qualifying = [
                row for row in retained if row["iou_target"] >= threshold
            ]
            variant_result[class_name] = {
                "all_correlation": correlation(
                    (row["decision_score"] for row in rows),
                    (row["iou_target"] for row in rows),
                ),
                "positive_correlation": correlation(
                    (row["decision_score"] for row in positive),
                    (row["iou_target"] for row in positive),
                ),
                "retained_count": len(retained),
                "threshold_purity": (
                    len(retained_qualifying) / len(retained) if retained else 0.0
                ),
                "threshold_coverage": (
                    len(retained_qualifying) / len(qualifying)
                    if qualifying else 0.0
                ),
                "deciles": decile_summary(rows),
            }
        result[variant] = variant_result
    return result


def audit_fixed_candidates(
    records: Sequence[Mapping[str, object]],
    variant_names: Sequence[str],
) -> None:
    """Require every variant to preserve the detector candidates exactly.

    The paper attributes changes to query-level decision evidence.  This
    audit therefore rejects an analysis whenever class predictions, the full
    semantic vector, decoded target IoUs, depth reliability, or any regressed
    geometry output differs across variants.
    """

    by_variant: Dict[str, Dict[Tuple[str, int], Mapping[str, object]]] = {}
    for name in variant_names:
        by_variant[name] = {
            (str(row["image_id"]), int(row["query_index"])): row
            for row in records if row["variant"] == name
        }
    reference_name = variant_names[0]
    reference = by_variant[reference_name]
    for name in variant_names[1:]:
        candidate = by_variant[name]
        if candidate.keys() != reference.keys():
            raise RuntimeError(
                "{} and {} do not contain identical queries".format(
                    reference_name, name
                )
            )
        for key, reference_row in reference.items():
            candidate_row = candidate[key]
            if int(candidate_row["class_index"]) != int(reference_row["class_index"]):
                raise RuntimeError("class candidate changed at {} for {}".format(key, name))
            for field in ("semantic_signature", "geometry_signature"):
                if candidate_row[field] != reference_row[field]:
                    raise RuntimeError(
                        "detector candidate field {} changed at {} for {}".format(
                            field, key, name
                        )
                    )
            for field in (
                "class_score", "depth_reliability", "iou_target",
            ):
                if not math.isclose(
                    float(candidate_row[field]),
                    float(reference_row[field]),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-7,
                ):
                    raise RuntimeError(
                        "detector candidate field {} changed at {} for {}".format(
                            field, key, name
                        )
                    )


def write_qualitative_case_index(
    output_dir: Path,
    records: Sequence[Mapping[str, object]],
    variant_names: Sequence[str],
) -> None:
    """Export candidate IDs for later success/failure visualization."""

    if len(variant_names) < 2:
        return
    first_name = variant_names[0]
    last_name = variant_names[-1]
    first = {
        (str(row["image_id"]), int(row["query_index"])): row
        for row in records if row["variant"] == first_name
    }
    last = {
        (str(row["image_id"]), int(row["query_index"])): row
        for row in records if row["variant"] == last_name
    }
    candidates = []
    for key, first_row in first.items():
        last_row = last[key]
        iou = float(first_row["iou_target"])
        if iou <= 0.0:
            continue
        first_error = abs(float(first_row["decision_score"]) - iou)
        last_error = abs(float(last_row["decision_score"]) - iou)
        candidates.append((first_error - last_error, key, first_row, last_row))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = (
        [("success", item) for item in candidates[:20]]
        + [("failure", item) for item in reversed(candidates[-20:])]
    )
    path = output_dir / "定性案例索引.csv"
    fields = [
        "case_type", "image_id", "query_index", "class_index", "iou_target",
        "baseline_score", "complete_score", "absolute_error_improvement",
        "box_x1", "box_y1", "box_x2", "box_y2",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case_type, (improvement, key, first_row, last_row) in selected:
            writer.writerow({
                "case_type": case_type,
                "image_id": key[0],
                "query_index": key[1],
                "class_index": first_row["class_index"],
                "iou_target": first_row["iou_target"],
                "baseline_score": first_row["decision_score"],
                "complete_score": last_row["decision_score"],
                "absolute_error_improvement": improvement,
                "box_x1": first_row["box_x1"],
                "box_y1": first_row["box_y1"],
                "box_x2": first_row["box_x2"],
                "box_y2": first_row["box_y2"],
            })


def run_variant(
    name: str,
    config_path: Path,
    checkpoint_path: Path,
    topk: int,
    max_batches: int,
) -> List[Dict[str, object]]:
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    cfg = load_config(config_path)
    cfg["dataset"]["batch_size"] = 1
    _, loader = build_qecr_dataloader(
        cfg["dataset"], workers=int(cfg["dataset"].get("workers", 8))
    )
    model, _ = build_model(cfg["model"])
    device = torch.device("cuda:0")
    model.to(device).eval()
    logger = logging.getLogger("MQD_metric_alignment")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    load_checkpoint(
        model=model,
        optimizer=None,
        filename=str(checkpoint_path),
        map_location=device,
        logger=logger,
        strict=False,
    )

    records: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch_index, (inputs, calibs, targets, info) in enumerate(loader):
            inputs = inputs.to(device)
            calibs = calibs.to(device)
            img_sizes = info["img_size_croped"].to(device)
            img_sizes_ori = info["img_size_original"].to(device)
            img_sizes_upper = info["upper"].to(device)
            outputs = model(
                inputs, calibs, targets, img_sizes, img_sizes_ori,
                img_sizes_upper, dn_args=0,
            )
            target_list = prepare_targets(targets, inputs.shape[0])
            target_outputs = dict(outputs)
            target_outputs.setdefault("quality_calibs", calibs)
            target_outputs.setdefault("quality_img_sizes", img_sizes)
            target_outputs.setdefault("quality_img_sizes_ori", img_sizes_ori)
            target_outputs.setdefault("quality_img_sizes_upper", img_sizes_upper)
            iou_targets = build_3d_iou_quality_targets(
                target_outputs, target_list
            ).detach().cpu()

            probabilities = outputs["pred_logits"].sigmoid()
            class_scores, labels = probabilities.max(dim=-1)
            depth_reliability = torch.exp(-outputs["pred_depth"][..., 1])
            evidence = outputs.get("pred_quality")
            if evidence is None:
                evidence = torch.ones_like(class_scores).unsqueeze(-1)
            evidence = evidence.squeeze(-1)
            decision_scores = class_scores * depth_reliability * evidence
            retained_count = min(
                int(topk), probabilities.shape[1] * probabilities.shape[2]
            )
            retained_pairs = torch.topk(
                probabilities.flatten(1), retained_count, dim=1
            ).indices.detach().cpu()
            retained_query_indices = retained_pairs // probabilities.shape[2]
            class_scores_cpu = class_scores.detach().cpu()
            labels_cpu = labels.detach().cpu()
            depth_reliability_cpu = depth_reliability.detach().cpu()
            evidence_cpu = evidence.detach().cpu()
            decision_scores_cpu = decision_scores.detach().cpu()
            encoded_boxes = outputs["pred_boxes"].detach().cpu()
            encoded_depth = outputs["pred_depth"].detach().cpu()
            encoded_dimensions = outputs["pred_3d_dim"].detach().cpu()
            encoded_angles = outputs["pred_angle"].detach().cpu()
            semantic_probabilities = probabilities.detach().cpu()

            for image_index in range(inputs.shape[0]):
                retained_set = set(retained_query_indices[image_index].tolist())
                raw_image_id = info["img_id"][image_index]
                image_id = str(
                    int(raw_image_id.item())
                    if torch.is_tensor(raw_image_id) else raw_image_id
                )
                for query_index in range(decision_scores.shape[1]):
                    box = encoded_boxes[image_index, query_index]
                    semantic_vector = semantic_probabilities[
                        image_index, query_index
                    ].reshape(-1)
                    geometry_vector = torch.cat([
                        box.reshape(-1),
                        encoded_depth[image_index, query_index].reshape(-1),
                        encoded_dimensions[image_index, query_index].reshape(-1),
                        encoded_angles[image_index, query_index].reshape(-1),
                    ])
                    records.append({
                        "variant": name,
                        "image_id": image_id,
                        "query_index": query_index,
                        "class_index": int(labels_cpu[image_index, query_index]),
                        "class_score": float(class_scores_cpu[image_index, query_index]),
                        "depth_reliability": float(depth_reliability_cpu[image_index, query_index]),
                        "metric_evidence": float(evidence_cpu[image_index, query_index]),
                        "decision_score": float(decision_scores_cpu[image_index, query_index]),
                        "iou_target": float(iou_targets[image_index, query_index]),
                        "retained": query_index in retained_set,
                        "semantic_signature": hashlib.sha256(
                            semantic_vector.contiguous().numpy().tobytes()
                        ).hexdigest(),
                        "geometry_signature": hashlib.sha256(
                            geometry_vector.contiguous().numpy().tobytes()
                        ).hexdigest(),
                        "box_x1": float(box[0] - box[2]),
                        "box_y1": float(box[1] - box[4]),
                        "box_x2": float(box[0] + box[3]),
                        "box_y2": float(box[1] + box[5]),
                    })
            if max_batches > 0 and batch_index + 1 >= max_batches:
                break
    return records


def write_outputs(
    output_dir: Path,
    records: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "查询级度量证据明细.csv"
    export_fields = [
        field for field in records[0].keys()
        if field not in ("semantic_signature", "geometry_signature")
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=export_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "度量一致性汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# MQD查询级度量一致性分析",
        "",
        "本报告固定检测候选与三维框，只比较不同决策证据和真实同类别最大3D IoU之间的关系。",
        "",
        "| 版本 | 类别 | 全部查询 Pearson/Spearman | 正IoU查询 Pearson/Spearman | Top-K阈值纯度 | Top-K阈值覆盖率 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant, class_results in summary.items():
        for class_name, values in class_results.items():
            all_corr = values["all_correlation"]
            positive_corr = values["positive_correlation"]
            def pair_text(item: Mapping[str, object]) -> str:
                if item["pearson"] is None:
                    return "—"
                return "{:.4f}/{:.4f}".format(
                    item["pearson"], item["spearman"]
                )
            lines.append(
                "| {} | {} | {} | {} | {:.2%} | {:.2%} |".format(
                    variant, class_name, pair_text(all_corr),
                    pair_text(positive_corr), values["threshold_purity"],
                    values["threshold_coverage"],
                )
            )
    lines.extend([
        "",
        "- `查询级度量证据明细.csv`保存逐查询可复核记录。",
        "- `度量一致性汇总.json`保存相关性、十等分曲线、纯度和覆盖率。",
        "- `定性案例索引.csv`给出固定候选上的成功与失败案例索引，可据此绘制三种证据并列图。",
        "- Top-K严格复现正式推理的查询—类别对选择，再映射到被保留的查询。",
        "- 跨版本候选类别、完整语义向量、深度可靠性、全部回归几何输出及IoU目标已经逐项审计为一致。",
        "- 该报告不是KITTI正式精度，不得替代官方AP结果。",
    ])
    (output_dir / "度量一致性报告.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)
    for axis, class_name in zip(axes, CLASS_NAMES):
        for variant, class_results in summary.items():
            deciles = class_results[class_name]["deciles"]
            axis.plot(
                [item["mean_score"] for item in deciles],
                [item["mean_iou"] for item in deciles],
                marker="o",
                linewidth=1.8,
                label=variant,
            )
        axis.set_title(class_name)
        axis.set_xlabel("Mean decision evidence")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean same-class 3D IoU")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "分数区间与三维IoU关系.png", dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MQD metric-alignment analysis")
    if args.topk <= 0:
        raise ValueError("topk must be positive")
    records: List[Dict[str, object]] = []
    variant_names: List[str] = []
    for spec in args.variant:
        parsed = parse_variant(spec)
        variant_names.append(parsed[0])
        records.extend(run_variant(*parsed, args.topk, args.max_batches))
    if not records:
        raise RuntimeError("no query records were generated")
    if len(set(variant_names)) != len(variant_names):
        raise ValueError("variant names must be unique")
    audit_fixed_candidates(records, variant_names)
    summary = summarize(records)
    output_dir = Path(args.output_dir)
    write_outputs(output_dir, records, summary)
    write_qualitative_case_index(output_dir, records, variant_names)
    print("MQD度量一致性分析完成：", Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
