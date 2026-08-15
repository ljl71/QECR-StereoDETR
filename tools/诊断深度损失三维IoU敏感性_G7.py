#!/usr/bin/env python3
"""G7：诊断米制深度损失与KITTI三维IoU目标之间的错位。

脚本只读取V09保存的验证集预测，不训练、不加载模型、不修改checkpoint。
它对同类别2D匹配预测实施小比例/小幅度GT深度Oracle修正，并比较：

1. 少量深度改善能否传递为正式KITTI 3D AP_R40；
2. 原始米制绝对深度误差、轴向尺寸归一化误差、轴向区间IoU损失，哪一个
   与完整3D IoU误差更一致；
3. 不同类别和距离段跨越正式3D IoU阈值的数量。

所有修正都使用验证集GT，只能作为训练损失立项诊断，不能作为模型结果。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


项目根目录 = Path(__file__).resolve().parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

from 诊断三维框组件误差_G5 import (  # noqa: E402
    一对一二维匹配,
    从投影中心恢复底面位置,
    保存标注目录,
    投影中心,
    深复制标注,
    评估结果目录,
    读取样本号,
)


类别 = ("Car", "Pedestrian", "Cyclist")
正式阈值 = {"Car": 0.70, "Pedestrian": 0.50, "Cyclist": 0.50}
变体参数 = {
    "GT残差10%": ("fraction", 0.10),
    "GT残差25%": ("fraction", 0.25),
    "GT残差50%": ("fraction", 0.50),
    "GT方向最大0.10m": ("cap", 0.10),
    "GT方向最大0.25m": ("cap", 0.25),
    "GT方向最大0.50m": ("cap", 0.50),
}


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G7：V09深度损失与三维IoU敏感性零训练诊断"
    )
    parser.add_argument(
        "--config",
        default="versions/V09_V08O加点式三维质量排序/config.yaml",
    )
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--label_dir", default=None)
    parser.add_argument("--calib_dir", default=None)
    parser.add_argument("--split_file", default=None)
    parser.add_argument(
        "--output_dir",
        default="outputs/深度损失三维IoU敏感性诊断_G7",
    )
    parser.add_argument("--match_iou_2d", type=float, default=0.30)
    parser.add_argument("--iou_chunk", type=int, default=256)
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def 轴向半投影(dimensions_lhw: np.ndarray, rotation_y: float) -> float:
    length, _, width = np.asarray(dimensions_lhw, dtype=np.float64).reshape(3)
    return 0.5 * (
        abs(float(length) * math.sin(float(rotation_y)))
        + abs(float(width) * math.cos(float(rotation_y)))
    )


def 一维区间IoU(
    center1: float,
    half1: float,
    center2: float,
    half2: float,
) -> float:
    left = max(float(center1) - float(half1), float(center2) - float(half2))
    right = min(float(center1) + float(half1), float(center2) + float(half2))
    intersection = max(right - left, 0.0)
    union = 2.0 * float(half1) + 2.0 * float(half2) - intersection
    return intersection / union if union > 0.0 else 0.0


def Oracle修正量(residual: float, mode: str, value: float) -> float:
    if mode == "fraction":
        return float(value) * float(residual)
    if mode == "cap":
        return float(np.clip(float(residual), -float(value), float(value)))
    raise KeyError(mode)


def 标注框(annotation: Mapping[str, np.ndarray], index: int) -> np.ndarray:
    location = np.asarray(annotation["location"][index], dtype=np.float64)
    dimensions = np.asarray(annotation["dimensions"][index], dtype=np.float64)
    return np.concatenate(
        [
            location,
            dimensions,
            np.asarray([annotation["rotation_y"][index]], dtype=np.float64),
        ]
    )


def 成对三维IoU(
    dt_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """调用项目正式KITTI 3D IoU实现，分块后只取一一对应的对角线。"""

    from lib.datasets.kitti.kitti_eval_python.eval import d3_box_overlap

    dt_boxes = np.asarray(dt_boxes, dtype=np.float64).reshape(-1, 7)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 7)
    if len(dt_boxes) != len(gt_boxes):
        raise ValueError("成对3D框数量不一致")
    values: List[np.ndarray] = []
    for start in range(0, len(dt_boxes), chunk_size):
        end = min(start + chunk_size, len(dt_boxes))
        matrix = d3_box_overlap(dt_boxes[start:end], gt_boxes[start:end])
        values.append(np.diag(matrix).astype(np.float64, copy=True))
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float64)


def 安全Spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.std(x) < 1.0e-12 or np.std(y) < 1.0e-12:
        return math.nan
    return float(spearmanr(x, y).correlation)


def 构造修正标注(
    dt_annos: Sequence[Mapping[str, np.ndarray]],
    gt_annos: Sequence[Mapping[str, np.ndarray]],
    all_matches: Sequence[Sequence[Tuple[int, int, float]]],
    calibs: Sequence[object],
    mode: str,
    value: float,
) -> List[MutableMapping[str, np.ndarray]]:
    corrected: List[MutableMapping[str, np.ndarray]] = []
    for dt, gt, matches, calib in zip(dt_annos, gt_annos, all_matches, calibs):
        anno = 深复制标注(dt)
        for dt_index, gt_index, _ in matches:
            class_name = str(dt["name"][dt_index])
            if class_name not in 类别:
                continue
            location = np.asarray(dt["location"][dt_index], dtype=np.float64)
            dimensions = np.asarray(dt["dimensions"][dt_index], dtype=np.float64)
            target_depth = float(gt["location"][gt_index][2])
            residual = target_depth - float(location[2])
            new_depth = float(location[2]) + Oracle修正量(residual, mode, value)
            if not math.isfinite(new_depth) or new_depth <= 0.1:
                continue
            uv = 投影中心(calib, location, dimensions)
            anno["location"][dt_index] = 从投影中心恢复底面位置(
                calib, uv, new_depth, dimensions
            )
        corrected.append(anno)
    return corrected


def 收集匹配元数据(
    gt_annos: Sequence[Mapping[str, np.ndarray]],
    dt_annos: Sequence[Mapping[str, np.ndarray]],
    all_matches: Sequence[Sequence[Tuple[int, int, float]]],
) -> Tuple[List[Dict[str, object]], np.ndarray, np.ndarray]:
    rows: List[Dict[str, object]] = []
    dt_boxes: List[np.ndarray] = []
    gt_boxes: List[np.ndarray] = []
    for gt, dt, matches in zip(gt_annos, dt_annos, all_matches):
        for dt_index, gt_index, iou2d in matches:
            class_name = str(dt["name"][dt_index])
            if class_name not in 类别:
                continue
            pred_location = np.asarray(dt["location"][dt_index], dtype=np.float64)
            gt_location = np.asarray(gt["location"][gt_index], dtype=np.float64)
            pred_dims = np.asarray(dt["dimensions"][dt_index], dtype=np.float64)
            gt_dims = np.asarray(gt["dimensions"][gt_index], dtype=np.float64)
            pred_yaw = float(dt["rotation_y"][dt_index])
            gt_yaw = float(gt["rotation_y"][gt_index])
            pred_half = 轴向半投影(pred_dims, pred_yaw)
            gt_half = 轴向半投影(gt_dims, gt_yaw)
            error = abs(float(pred_location[2]) - float(gt_location[2]))
            axial_iou = 一维区间IoU(
                float(pred_location[2]), pred_half,
                float(gt_location[2]), gt_half,
            )
            distance = float(gt_location[2])
            if distance < 20.0:
                distance_group = "近距_<20m"
            elif distance < 40.0:
                distance_group = "中距_20-40m"
            else:
                distance_group = "远距_>=40m"
            rows.append(
                {
                    "class": class_name,
                    "distance_group": distance_group,
                    "depth_error": error,
                    "normalized_depth_error": error / max(2.0 * gt_half, 0.1),
                    "axial_iou_loss": 1.0 - axial_iou,
                    "iou2d": float(iou2d),
                }
            )
            dt_boxes.append(标注框(dt, dt_index))
            gt_boxes.append(标注框(gt, gt_index))
    return rows, np.asarray(dt_boxes), np.asarray(gt_boxes)


def 收集变体配对框(
    annotations: Sequence[Mapping[str, np.ndarray]],
    gt_annos: Sequence[Mapping[str, np.ndarray]],
    all_matches: Sequence[Sequence[Tuple[int, int, float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    dt_boxes: List[np.ndarray] = []
    gt_boxes: List[np.ndarray] = []
    for dt, gt, matches in zip(annotations, gt_annos, all_matches):
        for dt_index, gt_index, _ in matches:
            if str(dt["name"][dt_index]) not in 类别:
                continue
            dt_boxes.append(标注框(dt, dt_index))
            gt_boxes.append(标注框(gt, gt_index))
    return np.asarray(dt_boxes), np.asarray(gt_boxes)


def 连续敏感性汇总(
    rows: Sequence[Mapping[str, object]],
    current_iou: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    if len(rows) != len(current_iou):
        raise ValueError("匹配元数据与3D IoU数量不一致")
    groups: Dict[str, np.ndarray] = {
        "全部": np.ones(len(rows), dtype=bool),
    }
    for class_name in 类别:
        groups[class_name] = np.asarray(
            [row["class"] == class_name for row in rows], dtype=bool
        )
    for distance_group in ("近距_<20m", "中距_20-40m", "远距_>=40m"):
        groups[f"Car_{distance_group}"] = np.asarray(
            [
                row["class"] == "Car" and row["distance_group"] == distance_group
                for row in rows
            ],
            dtype=bool,
        )

    depth_error = np.asarray([row["depth_error"] for row in rows], dtype=np.float64)
    normalized = np.asarray(
        [row["normalized_depth_error"] for row in rows], dtype=np.float64
    )
    axial_loss = np.asarray([row["axial_iou_loss"] for row in rows], dtype=np.float64)
    full_loss = 1.0 - np.asarray(current_iou, dtype=np.float64)
    summary: Dict[str, Dict[str, float]] = {}
    for name, mask in groups.items():
        count = int(mask.sum())
        summary[name] = {
            "count": count,
            "depth_mae_m": float(np.mean(depth_error[mask])) if count else math.nan,
            "normalized_depth_error_mean": float(np.mean(normalized[mask])) if count else math.nan,
            "axial_iou_mean": float(np.mean(1.0 - axial_loss[mask])) if count else math.nan,
            "full_3d_iou_mean": float(np.mean(current_iou[mask])) if count else math.nan,
            "spearman_raw_to_full_loss": 安全Spearman(depth_error[mask], full_loss[mask]),
            "spearman_normalized_to_full_loss": 安全Spearman(normalized[mask], full_loss[mask]),
            "spearman_axial_to_full_loss": 安全Spearman(axial_loss[mask], full_loss[mask]),
        }
    return summary


def 阈值跨越汇总(
    rows: Sequence[Mapping[str, object]],
    current_iou: np.ndarray,
    variant_ious: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    classes = np.asarray([row["class"] for row in rows])
    for variant, values in variant_ious.items():
        result[variant] = {}
        for class_name in 类别:
            mask = classes == class_name
            threshold = 正式阈值[class_name]
            before = np.asarray(current_iou)[mask]
            after = np.asarray(values)[mask]
            crossed_up = int(np.sum((before < threshold) & (after >= threshold)))
            crossed_down = int(np.sum((before >= threshold) & (after < threshold)))
            near = (before >= threshold - 0.10) & (before < threshold)
            near_recovered = int(np.sum(near & (after >= threshold)))
            result[variant][class_name] = {
                "count": int(mask.sum()),
                "crossed_up": crossed_up,
                "crossed_down": crossed_down,
                "net_crossing": crossed_up - crossed_down,
                "near_below_count": int(near.sum()),
                "near_recovered": near_recovered,
                "near_recovery_rate": (
                    float(near_recovered / near.sum()) if near.sum() else math.nan
                ),
            }
    return result


def 三元组(metrics: Mapping[str, Mapping[str, float]], class_name: str) -> str:
    item = metrics[class_name]
    return f"{item['Easy']:.4f}/{item['Moderate']:.4f}/{item['Hard']:.4f}"


def 自动判定(
    baseline: Mapping[str, Mapping[str, float]],
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    sensitivity: Mapping[str, Mapping[str, float]],
    crossings: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> Tuple[str, str]:
    primary_names = ("GT残差10%", "GT方向最大0.10m")
    best = max(
        primary_names,
        key=lambda name: metrics[name]["Car"]["Moderate"] - baseline["Car"]["Moderate"],
    )
    gain = metrics[best]["Car"]["Moderate"] - baseline["Car"]["Moderate"]
    car_protected = all(
        metrics[best]["Car"][level] - baseline["Car"][level] >= -0.10
        for level in ("Easy", "Hard")
    )
    small_classes_protected = all(
        metrics[best][name][level] - baseline[name][level] >= -0.10
        for name in ("Pedestrian", "Cyclist")
        for level in ("Moderate", "Hard")
    )
    car_stats = sensitivity["Car"]
    correlation_gain = (
        car_stats["spearman_axial_to_full_loss"]
        - car_stats["spearman_raw_to_full_loss"]
    )
    net_crossing = crossings[best]["Car"]["net_crossing"]
    if (
        gain >= 0.20
        and car_protected
        and small_classes_protected
        and correlation_gain >= 0.03
        and net_crossing >= 20
    ):
        return (
            "通过，允许实现V22训练期轴向IoU敏感深度损失",
            f"{best}使Car Moderate提升{gain:+.4f} AP、净跨阈值{net_crossing}个，轴向损失相关性比米制误差提高{correlation_gain:+.4f}。",
        )
    if gain >= 0.20 and car_protected and small_classes_protected:
        return (
            "深度敏感性通过，但轴向IoU损失证据不足",
            f"{best}的Car Moderate提升{gain:+.4f} AP，但轴向损失相关性增量{correlation_gain:+.4f}或净跨阈值数{net_crossing}未通过；不得直接实现V22。",
        )
    return (
        "否决，不开发训练期轴向IoU深度损失",
        f"最佳小修正{best}的Car Moderate变化{gain:+.4f} AP，未达到+0.20 AP门槛或触发类别保护线。",
    )


def 写报告(
    path: Path,
    baseline: Mapping[str, Mapping[str, float]],
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    sensitivity: Mapping[str, Mapping[str, float]],
    crossings: Mapping[str, Mapping[str, Mapping[str, float]]],
    matched_count: int,
    conclusion: Tuple[str, str],
    elapsed: float,
) -> None:
    lines = [
        "# StereoDETR深度损失三维IoU敏感性G7诊断报告",
        "",
        "## 结论先行",
        "",
        f"- 自动结论：**{conclusion[0]}**。{conclusion[1]}",
        "- 所有修正都使用验证集GT方向或残差，只是Oracle敏感性，不是模型结果。",
        "- 本诊断不训练、不加载模型、不修改V09权重；V09原始预测始终保留。",
        "",
        "## 实验设置",
        "",
        f"- 同类别2D一对一匹配数：`{matched_count}`。",
        "- 比例修正只沿预测投影射线消除10%/25%/50%的真实轴向深度残差。",
        "- 限幅修正只使用真实方向，每个对象最多移动0.10/0.25/0.50米。",
        f"- 总耗时：`{elapsed/60.0:.2f}`分钟。",
        "",
        "## 完整验证集KITTI 3D AP_R40",
        "",
        "| 方法 | Car E/M/H | ΔCar M | Pedestrian E/M/H | Cyclist E/M/H |",
        "|---|---:|---:|---:|---:|",
        f"| V09原始预测 | {三元组(baseline, 'Car')} | +0.0000 | {三元组(baseline, 'Pedestrian')} | {三元组(baseline, 'Cyclist')} |",
    ]
    for variant in 变体参数:
        item = metrics[variant]
        delta = item["Car"]["Moderate"] - baseline["Car"]["Moderate"]
        lines.append(
            f"| {variant} | {三元组(item, 'Car')} | {delta:+.4f} | "
            f"{三元组(item, 'Pedestrian')} | {三元组(item, 'Cyclist')} |"
        )

    lines.extend(
        [
            "",
            "## 深度误差与完整3D IoU损失的相关性",
            "",
            "| 分组 | 数量 | 深度MAE(m) | 归一化误差 | 轴向IoU | 完整3DIoU | 米制误差Spearman | 归一化Spearman | 轴向损失Spearman |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("全部", "Car", "Pedestrian", "Cyclist", "Car_近距_<20m", "Car_中距_20-40m", "Car_远距_>=40m"):
        item = sensitivity[name]
        lines.append(
            f"| {name} | {item['count']} | {item['depth_mae_m']:.4f} | "
            f"{item['normalized_depth_error_mean']:.4f} | {item['axial_iou_mean']:.4f} | "
            f"{item['full_3d_iou_mean']:.4f} | {item['spearman_raw_to_full_loss']:+.4f} | "
            f"{item['spearman_normalized_to_full_loss']:+.4f} | {item['spearman_axial_to_full_loss']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 成对匹配框的正式IoU阈值跨越",
            "",
            "| 方法/类别 | 负→正 | 正→负 | 净跨越 | 阈值下0.10内数量 | 恢复数/比例 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in 变体参数:
        for class_name in 类别:
            item = crossings[variant][class_name]
            lines.append(
                f"| {variant}/{class_name} | {item['crossed_up']} | {item['crossed_down']} | "
                f"{item['net_crossing']:+d} | {item['near_below_count']} | "
                f"{item['near_recovered']}/{100.0*item['near_recovery_rate']:.2f}% |"
            )

    lines.extend(
        [
            "",
            "## 预注册门槛",
            "",
            "1. 主候选仅允许比较GT残差10%和GT方向最大0.10m，不从更强Oracle中事后挑最好值。",
            "2. Car Moderate至少提升0.20 AP；Car Easy/Hard及两个小类别Moderate/Hard均不得下降0.10 AP以上。",
            "3. Car轴向IoU损失与完整3D IoU损失的Spearman必须比原始米制误差至少提高0.03。",
            "4. 主候选在Car匹配框上的净跨正式0.70阈值数量至少20。",
            "5. 通过只允许实现训练期、零推理开销的V22五轮公平配对；不直接授权195轮。",
            "",
            "## 解释边界",
            "",
            "- Oracle修正有效，只说明现有框对小幅深度改善敏感，不证明网络能预测GT修正方向。",
            "- 轴向损失相关性通过，才支持把统一米制L1补充为尺寸/朝向敏感训练目标。",
            "- 正式V22若立项，只能使用训练集GT计算辅助损失；推理结构、参数量和时延必须与V09一致。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def 转JSON安全(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): 转JSON安全(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [转JSON安全(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def 自检() -> None:
    dims = np.asarray([4.0, 2.0, 2.0])
    assert abs(轴向半投影(dims, 0.0) - 1.0) < 1.0e-9
    assert abs(轴向半投影(dims, math.pi / 2.0) - 2.0) < 1.0e-9
    assert abs(一维区间IoU(0.0, 1.0, 0.0, 1.0) - 1.0) < 1.0e-9
    assert abs(一维区间IoU(0.0, 1.0, 1.0, 1.0) - 1.0 / 3.0) < 1.0e-9
    assert abs(Oracle修正量(2.0, "fraction", 0.1) - 0.2) < 1.0e-9
    assert abs(Oracle修正量(-2.0, "cap", 0.25) + 0.25) < 1.0e-9
    print("G7自检通过：轴向半投影、区间IoU、比例修正与限幅修正正常")


def 主程序(args: argparse.Namespace) -> None:
    from lib.datasets.kitti.kitti_utils import Calibration
    from lib.helpers.config_helper import load_config

    if not 0.0 <= args.match_iou_2d <= 1.0:
        raise ValueError("match_iou_2d必须位于[0,1]")
    if args.iou_chunk <= 0:
        raise ValueError("iou_chunk必须大于0")
    start = time.time()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    trainer_cfg = cfg["trainer"]
    results_dir = Path(args.results_dir) if args.results_dir else (
        Path(trainer_cfg["save_path"]) / cfg["model_name"] / "outputs" / "data"
    )
    label_dir = Path(args.label_dir) if args.label_dir else Path(dataset_cfg["root_dir_eval"]) / "label_2"
    calib_dir = Path(args.calib_dir) if args.calib_dir else Path(dataset_cfg["root_dir_eval"]) / "calib"
    split_file = Path(args.split_file) if args.split_file else Path(dataset_cfg["eval_txt"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, label in (
        (results_dir, "V09预测目录"),
        (label_dir, "KITTI标签目录"),
        (calib_dir, "KITTI标定目录"),
        (split_file, "验证划分文件"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"找不到{label}：{path}")

    sample_ids = 读取样本号(split_file)
    baseline, gt_annos, dt_annos = 评估结果目录(
        results_dir, label_dir, sample_ids
    )
    calibs = [Calibration(str(calib_dir / f"{sample_id}.txt")) for sample_id in sample_ids]
    all_matches = [
        一对一二维匹配(gt, dt, args.match_iou_2d)
        for gt, dt in zip(gt_annos, dt_annos)
    ]
    matched_count = sum(len(matches) for matches in all_matches)
    if matched_count < 3000:
        raise RuntimeError(f"仅匹配{matched_count}个对象，低于诊断下限3000")

    rows, current_dt_boxes, pair_gt_boxes = 收集匹配元数据(
        gt_annos, dt_annos, all_matches
    )
    current_iou = 成对三维IoU(
        current_dt_boxes, pair_gt_boxes, args.iou_chunk
    )
    sensitivity = 连续敏感性汇总(rows, current_iou)

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    variant_ious: Dict[str, np.ndarray] = {}
    for variant, (mode, value) in 变体参数.items():
        corrected = 构造修正标注(
            dt_annos, gt_annos, all_matches, calibs, mode, value
        )
        data_dir = output_dir / "Oracle_仅诊断_禁止提交" / variant / "data"
        保存标注目录(corrected, sample_ids, data_dir)
        metrics[variant], _, _ = 评估结果目录(data_dir, label_dir, sample_ids)
        variant_dt_boxes, variant_gt_boxes = 收集变体配对框(
            corrected, gt_annos, all_matches
        )
        variant_ious[variant] = 成对三维IoU(
            variant_dt_boxes, variant_gt_boxes, args.iou_chunk
        )

    crossings = 阈值跨越汇总(rows, current_iou, variant_ious)
    conclusion = 自动判定(baseline, metrics, sensitivity, crossings)
    summary = {
        "限制": "全部修正使用验证集GT，仅为Oracle敏感性，禁止作为模型结果或KITTI提交",
        "paths": {
            "results_dir": str(results_dir),
            "label_dir": str(label_dir),
            "calib_dir": str(calib_dir),
            "split_file": str(split_file),
        },
        "matched_count": matched_count,
        "baseline": baseline,
        "oracle_metrics": metrics,
        "sensitivity": sensitivity,
        "threshold_crossings": crossings,
        "conclusion": {"level": conclusion[0], "explanation": conclusion[1]},
    }
    (output_dir / "诊断汇总.json").write_text(
        json.dumps(转JSON安全(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = output_dir / "诊断报告.md"
    写报告(
        report,
        baseline,
        metrics,
        sensitivity,
        crossings,
        matched_count,
        conclusion,
        time.time() - start,
    )
    print("G7诊断完成")
    print("报告:", report)
    print("自动结论:", conclusion[0])
    print("解释:", conclusion[1])


def main() -> None:
    args = 解析参数()
    if args.self_test:
        自检()
        return
    主程序(args)


if __name__ == "__main__":
    main()
