#!/usr/bin/env python3
"""G6：诊断 StereoDETR 的几何深度公式及其朝向相关偏差。

本脚本不训练、不加载模型、不修改 checkpoint。它只读取 V09 已保存的 KITTI
验证集预测，把样本按编号奇偶固定拆成校准集和留出集。校准集只拟合每类别一个
标量融合系数；所有结论和自动门槛只看从未参与拟合的留出集。

注意：校准仍使用了验证集真值，因此本脚本输出只能作为诊断证据，不能作为模型
精度、论文主表结果或 KITTI 提交结果。正式版本必须在训练集上学习并在完整验证集
重新评估。
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
方法 = (
    "无半长补偿",
    "当前固定半长",
    "朝向感知半轴",
    "仅修正固定半长偏差",
)


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G6：V09几何深度公式与朝向相关偏差的校准/留出诊断"
    )
    parser.add_argument(
        "--config",
        default="versions/V09_V08O加点式三维质量排序/config.yaml",
        help="只用于推导 V09 输出及 KITTI 路径",
    )
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--label_dir", default=None)
    parser.add_argument("--calib_dir", default=None)
    parser.add_argument("--split_file", default=None)
    parser.add_argument(
        "--output_dir",
        default="outputs/朝向感知几何深度诊断_G6",
    )
    parser.add_argument("--match_iou_2d", type=float, default=0.30)
    parser.add_argument(
        "--alpha_limit",
        type=float,
        default=1.0,
        help="校准标量的绝对值上限，默认预注册为1.0",
    )
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def 几何深度候选(
    focal: float,
    bbox_height: float,
    dimensions_lhw: np.ndarray,
    rotation_y: float,
) -> Dict[str, float]:
    """复现当前公式，并构造无半长与朝向感知两个可检验候选。

    项目从 KITTI 文本读入后的尺寸顺序为 [l,h,w]。当前网络公式等价于
    ``f*h/box_h + l/2``。对 KITTI 绕相机 Y 轴旋转的三维框，其 Z 轴半投影
    应为 ``(|l*sin(ry)| + |w*cos(ry)|)/2``。
    """

    dims = np.asarray(dimensions_lhw, dtype=np.float64).reshape(3)
    length, height, width = [float(value) for value in dims]
    base = float(focal) * height / max(float(bbox_height), 1.0)
    fixed_half = 0.5 * length
    yaw_half = 0.5 * (
        abs(length * math.sin(float(rotation_y)))
        + abs(width * math.cos(float(rotation_y)))
    )
    return {
        "无半长补偿": base,
        "当前固定半长": base + fixed_half,
        "朝向感知半轴": base + yaw_half,
        # 这一项只测试把现有固定半长替换为朝向半轴所提供的增量信号。
        "仅修正固定半长偏差": yaw_half - fixed_half,
    }


def 最小二乘标量(gap: np.ndarray, target: np.ndarray, limit: float) -> float:
    gap = np.asarray(gap, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(gap) & np.isfinite(target)
    gap = gap[valid]
    target = target[valid]
    denominator = float(np.dot(gap, gap))
    if len(gap) < 2 or denominator <= 1.0e-12:
        return 0.0
    value = float(np.dot(gap, target) / denominator)
    return float(np.clip(value, -abs(limit), abs(limit)))


def 安全相关性(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.std(x) < 1.0e-12 or np.std(y) < 1.0e-12:
        return math.nan, math.nan
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(spearmanr(x, y).correlation)
    return pearson, spearman


def 样本属于校准集(sample_id: str) -> bool:
    # 固定规则，避免看完结果后反复换划分。
    return int(sample_id) % 2 == 0


def 收集匹配记录(
    sample_ids: Sequence[str],
    gt_annos: Sequence[Mapping[str, np.ndarray]],
    dt_annos: Sequence[Mapping[str, np.ndarray]],
    calibs: Sequence[object],
    threshold: float,
) -> Tuple[List[Dict[str, object]], List[List[Tuple[int, int, float]]]]:
    rows: List[Dict[str, object]] = []
    all_matches: List[List[Tuple[int, int, float]]] = []
    for sample_id, gt, dt, calib in zip(sample_ids, gt_annos, dt_annos, calibs):
        matches = 一对一二维匹配(gt, dt, threshold)
        all_matches.append(matches)
        focal = float(calib.P2[0, 0])
        for dt_index, gt_index, iou2d in matches:
            class_name = str(dt["name"][dt_index])
            if class_name not in 类别:
                continue
            bbox = np.asarray(dt["bbox"][dt_index], dtype=np.float64)
            dims = np.asarray(dt["dimensions"][dt_index], dtype=np.float64)
            pred_depth = float(dt["location"][dt_index][2])
            gt_depth = float(gt["location"][gt_index][2])
            candidates = 几何深度候选(
                focal,
                float(bbox[3] - bbox[1]),
                dims,
                float(dt["rotation_y"][dt_index]),
            )
            gaps = {
                name: (
                    float(value)
                    if name == "仅修正固定半长偏差"
                    else float(value) - pred_depth
                )
                for name, value in candidates.items()
            }
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": "calibration" if 样本属于校准集(sample_id) else "holdout",
                    "class": class_name,
                    "dt_index": int(dt_index),
                    "gt_index": int(gt_index),
                    "iou2d": float(iou2d),
                    "pred_depth": pred_depth,
                    "gt_depth": gt_depth,
                    "target_residual": gt_depth - pred_depth,
                    "gaps": gaps,
                }
            )
    return rows, all_matches


def 拟合分类标量(
    rows: Sequence[Mapping[str, object]],
    method: str,
    limit: float,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    calibration = [row for row in rows if row["split"] == "calibration"]
    for class_name in 类别:
        selected = [row for row in calibration if row["class"] == class_name]
        result[class_name] = 最小二乘标量(
            np.asarray([row["gaps"][method] for row in selected]),
            np.asarray([row["target_residual"] for row in selected]),
            limit,
        )
    return result


def 连续误差统计(
    rows: Sequence[Mapping[str, object]],
    method: str,
    alphas: Mapping[str, float],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    holdout = [row for row in rows if row["split"] == "holdout"]
    for class_name in 类别:
        selected = [row for row in holdout if row["class"] == class_name]
        target = np.asarray([row["target_residual"] for row in selected], dtype=np.float64)
        gap = np.asarray([row["gaps"][method] for row in selected], dtype=np.float64)
        correction = float(alphas[class_name]) * gap
        before = np.abs(target)
        after = np.abs(target - correction)
        pearson, spearman = 安全相关性(gap, target)
        directional_mask = np.abs(target) >= 0.10
        direction = (
            float(np.mean(np.sign(correction[directional_mask]) == np.sign(target[directional_mask])))
            if np.any(directional_mask)
            else math.nan
        )
        result[class_name] = {
            "count": int(len(selected)),
            "alpha": float(alphas[class_name]),
            "mae_before": float(np.mean(before)) if len(before) else math.nan,
            "mae_after": float(np.mean(after)) if len(after) else math.nan,
            "mae_improvement_pct": (
                100.0 * (float(np.mean(before)) - float(np.mean(after))) / max(float(np.mean(before)), 1.0e-12)
                if len(before)
                else math.nan
            ),
            "pearson_gap_target": pearson,
            "spearman_gap_target": spearman,
            "direction_accuracy": direction,
        }
    return result


def 应用校准修正(
    dt_annos: Sequence[Mapping[str, np.ndarray]],
    calibs: Sequence[object],
    method: str,
    alphas: Mapping[str, float],
) -> List[MutableMapping[str, np.ndarray]]:
    corrected: List[MutableMapping[str, np.ndarray]] = []
    for dt, calib in zip(dt_annos, calibs):
        anno = 深复制标注(dt)
        focal = float(calib.P2[0, 0])
        for index in range(len(anno["name"])):
            class_name = str(anno["name"][index])
            if class_name not in alphas:
                continue
            bbox = np.asarray(anno["bbox"][index], dtype=np.float64)
            dims = np.asarray(anno["dimensions"][index], dtype=np.float64)
            location = np.asarray(anno["location"][index], dtype=np.float64)
            candidates = 几何深度候选(
                focal,
                float(bbox[3] - bbox[1]),
                dims,
                float(anno["rotation_y"][index]),
            )
            if method == "仅修正固定半长偏差":
                gap = candidates[method]
            else:
                gap = candidates[method] - float(location[2])
            new_depth = float(location[2]) + float(alphas[class_name]) * float(gap)
            if not math.isfinite(new_depth) or new_depth <= 0.1:
                continue
            uv = 投影中心(calib, location, dims)
            anno["location"][index] = 从投影中心恢复底面位置(
                calib, uv, new_depth, dims
            )
        corrected.append(anno)
    return corrected


def 三元组(metrics: Mapping[str, Mapping[str, float]], class_name: str) -> str:
    item = metrics[class_name]
    return f"{item['Easy']:.4f}/{item['Moderate']:.4f}/{item['Hard']:.4f}"


def 自动判定(
    baseline: Mapping[str, Mapping[str, float]],
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    continuous: Mapping[str, Mapping[str, Mapping[str, float]]],
    calibration_count: int,
    holdout_count: int,
) -> Tuple[str, str]:
    candidates = ("无半长补偿", "朝向感知半轴", "仅修正固定半长偏差")
    best = max(
        candidates,
        key=lambda name: metrics[name]["Car"]["Moderate"] - baseline["Car"]["Moderate"],
    )
    gain = metrics[best]["Car"]["Moderate"] - baseline["Car"]["Moderate"]
    fixed_gain = (
        metrics["当前固定半长"]["Car"]["Moderate"]
        - baseline["Car"]["Moderate"]
    )
    class_protected = all(
        metrics[best][name][level] - baseline[name][level] >= limit
        for name, limit in (("Pedestrian", -0.25), ("Cyclist", -0.25))
        for level in ("Moderate", "Hard")
    )
    car_protected = all(
        metrics[best]["Car"][level] - baseline["Car"][level] >= -0.10
        for level in ("Easy", "Hard")
    )
    enough = calibration_count >= 3000 and holdout_count >= 3000
    mae_gain = continuous[best]["Car"]["mae_improvement_pct"]
    if enough and gain >= 0.20 and car_protected and class_protected and mae_gain >= 1.0:
        relation = "优于当前固定公式" if gain >= fixed_gain + 0.10 else "与当前固定公式接近"
        return (
            "通过，允许开发V21短程配对版本",
            f"{best}在留出集Car Moderate提升{gain:+.4f} AP，Car深度MAE改善{mae_gain:+.2f}%，且{relation}。",
        )
    if enough and gain >= 0.10 and car_protected:
        return (
            "谨慎，只允许补充诊断，不开发正式版本",
            f"最佳候选{best}仅提升{gain:+.4f} AP，未完整通过0.20 AP、多类别保护和1% MAE门槛。",
        )
    return (
        "否决，停止几何深度公式路线",
        f"最佳候选{best}在留出集Car Moderate变化{gain:+.4f} AP，未证明可实现收益。",
    )


def 写报告(
    path: Path,
    baseline: Mapping[str, Mapping[str, float]],
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    alphas: Mapping[str, Mapping[str, float]],
    continuous: Mapping[str, Mapping[str, Mapping[str, float]]],
    calibration_count: int,
    holdout_count: int,
    conclusion: Tuple[str, str],
    elapsed: float,
) -> None:
    lines = [
        "# StereoDETR朝向感知几何深度G6诊断报告",
        "",
        "## 结论先行",
        "",
        f"- 自动结论：**{conclusion[0]}**。{conclusion[1]}",
        "- 本实验不训练、不加载模型、不修改V09权重；校准集与留出集按样本号奇偶固定隔离。",
        "- 校准使用了验证集真值，所有结果只能用于决定是否开发V21，不能写成模型精度。",
        "",
        "## 公式与诊断设置",
        "",
        "- 当前公式：`z_geo = f*h/box_h + l/2`。",
        "- 朝向感知候选：`z_geo = f*h/box_h + (|l*sin(ry)|+|w*cos(ry)|)/2`。",
        "- 每个类别、每种候选只在校准集拟合一个裁剪到[-1,1]的最小二乘标量。",
        f"- 校准匹配数：`{calibration_count}`；留出匹配数：`{holdout_count}`；总耗时：`{elapsed/60.0:.2f}`分钟。",
        "",
        "## 留出集KITTI 3D AP_R40",
        "",
        "| 方法 | Car E/M/H | ΔCar M | Pedestrian E/M/H | Cyclist E/M/H |",
        "|---|---:|---:|---:|---:|",
        f"| V09原始预测 | {三元组(baseline, 'Car')} | +0.0000 | {三元组(baseline, 'Pedestrian')} | {三元组(baseline, 'Cyclist')} |",
    ]
    for method in 方法:
        item = metrics[method]
        gain = item["Car"]["Moderate"] - baseline["Car"]["Moderate"]
        lines.append(
            f"| {method} | {三元组(item, 'Car')} | {gain:+.4f} | "
            f"{三元组(item, 'Pedestrian')} | {三元组(item, 'Cyclist')} |"
        )

    lines.extend(
        [
            "",
            "## 校准标量与留出集连续误差",
            "",
            "| 方法/类别 | alpha | 数量 | 深度MAE前→后(m) | MAE变化 | Pearson | Spearman | 方向正确率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in 方法:
        for class_name in 类别:
            item = continuous[method][class_name]
            lines.append(
                f"| {method}/{class_name} | {alphas[method][class_name]:+.5f} | {item['count']} | "
                f"{item['mae_before']:.4f}→{item['mae_after']:.4f} | "
                f"{item['mae_improvement_pct']:+.2f}% | {item['pearson_gap_target']:+.4f} | "
                f"{item['spearman_gap_target']:+.4f} | {100.0*item['direction_accuracy']:.2f}% |"
            )

    lines.extend(
        [
            "",
            "## 预注册判定线",
            "",
            "1. 校准、留出匹配数均不少于3000。",
            "2. 最佳可实现候选的留出集Car Moderate至少提升0.20 AP，Car深度MAE至少改善1%。",
            "3. Car Easy/Hard均不得下降0.10 AP以上；Pedestrian/Cyclist Moderate、Hard均不得下降0.25 AP以上。",
            "4. 通过也只允许开发同起点、同种子的V21短程配对版；完整训练仍需第二随机种子和同步CUDA时延复测。",
            "",
            "## 为什么该诊断能解释V20",
            "",
            "- 若朝向感知候选明显优于固定半长，说明V20输入的几何先验存在朝向相关系统误差，应先修公式再训练残差头。",
            "- 若所有候选都与真实残差低相关且AP无收益，说明V20失败主要不是头容量，而是现有二维框高与三维尺寸不足以提供可靠深度修正信号。",
            "- 若固定公式的简单标量有效而V20无效，才说明V20残差头或优化方式比输入先验更值得检查。",
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
    yaw0 = 几何深度候选(100.0, 10.0, dims, 0.0)
    yaw90 = 几何深度候选(100.0, 10.0, dims, math.pi / 2.0)
    assert abs(yaw0["无半长补偿"] - 20.0) < 1.0e-9
    assert abs(yaw0["当前固定半长"] - 22.0) < 1.0e-9
    assert abs(yaw0["朝向感知半轴"] - 21.0) < 1.0e-9
    assert abs(yaw90["朝向感知半轴"] - 22.0) < 1.0e-9
    gap = np.asarray([1.0, 2.0, 3.0])
    target = 0.5 * gap
    assert abs(最小二乘标量(gap, target, 1.0) - 0.5) < 1.0e-9
    assert 样本属于校准集("000002")
    assert not 样本属于校准集("000003")
    print("G6自检通过：尺寸顺序、朝向半轴、固定奇偶划分与标量拟合正常")


def 主程序(args: argparse.Namespace) -> None:
    from lib.datasets.kitti.kitti_utils import Calibration
    from lib.helpers.config_helper import load_config

    if not 0.0 <= args.match_iou_2d <= 1.0:
        raise ValueError("match_iou_2d必须位于[0,1]")
    if args.alpha_limit <= 0.0:
        raise ValueError("alpha_limit必须大于0")
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
    calibration_ids = [sample_id for sample_id in sample_ids if 样本属于校准集(sample_id)]
    holdout_ids = [sample_id for sample_id in sample_ids if not 样本属于校准集(sample_id)]
    _, gt_annos, dt_annos = 评估结果目录(results_dir, label_dir, sample_ids)
    calibs = [Calibration(str(calib_dir / f"{sample_id}.txt")) for sample_id in sample_ids]
    rows, _ = 收集匹配记录(
        sample_ids, gt_annos, dt_annos, calibs, args.match_iou_2d
    )
    calibration_count = sum(row["split"] == "calibration" for row in rows)
    holdout_count = sum(row["split"] == "holdout" for row in rows)
    if calibration_count < 100 or holdout_count < 100:
        raise RuntimeError(
            f"匹配不足：校准{calibration_count}，留出{holdout_count}；检查路径或2D IoU阈值"
        )

    alphas = {
        method: 拟合分类标量(rows, method, args.alpha_limit)
        for method in 方法
    }
    continuous = {
        method: 连续误差统计(rows, method, alphas[method])
        for method in 方法
    }

    id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    holdout_indices = [id_to_index[sample_id] for sample_id in holdout_ids]
    holdout_dt = [dt_annos[index] for index in holdout_indices]
    holdout_calibs = [calibs[index] for index in holdout_indices]
    baseline, _, _ = 评估结果目录(results_dir, label_dir, holdout_ids)
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in 方法:
        corrected = 应用校准修正(
            holdout_dt, holdout_calibs, method, alphas[method]
        )
        data_dir = output_dir / "仅诊断_禁止提交" / method / "data"
        保存标注目录(corrected, holdout_ids, data_dir)
        metrics[method], _, _ = 评估结果目录(data_dir, label_dir, holdout_ids)

    conclusion = 自动判定(
        baseline, metrics, continuous, calibration_count, holdout_count
    )
    summary = {
        "限制": "使用验证集偶数样本校准、奇数样本留出，只能诊断，禁止作为模型结果",
        "paths": {
            "results_dir": str(results_dir),
            "label_dir": str(label_dir),
            "calib_dir": str(calib_dir),
            "split_file": str(split_file),
        },
        "sample_count": {
            "all": len(sample_ids),
            "calibration": len(calibration_ids),
            "holdout": len(holdout_ids),
            "calibration_matches": calibration_count,
            "holdout_matches": holdout_count,
        },
        "alphas": alphas,
        "continuous_holdout": continuous,
        "baseline_holdout": baseline,
        "corrected_holdout": metrics,
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
        alphas,
        continuous,
        calibration_count,
        holdout_count,
        conclusion,
        time.time() - start,
    )
    print("G6诊断完成")
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
