#!/usr/bin/env python3
"""V12-G0：诊断 StereoDETR 相关体在全通道平均前丢失的信息。

该脚本不训练、不修改 checkpoint，也不把 Velodyne 输入检测器。它仅在离线
诊断中把 KITTI 激光点投影成稀疏真实视差，读取 V09 冻结网络送入
LightStereo 深度预测器的三层左右特征，并比较：

1. 当前实现的全通道平均相关（等价于各组相关的均匀平均）；
2. 无参数的逐视差分组最大值；
3. 逐点选择最优通道组的 Oracle 上限。

Oracle 使用真值，只用于判断“额外信息是否存在”，绝不是模型结果。只有正式
样本达到预注册门槛，才值得实现零初始化、压回原 D 通道的轻量分组门控。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


项目根目录 = Path(__file__).resolve().parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))


方法顺序 = ("当前均匀平均", "分组最大聚合", "逐点最优分组_Oracle")
分组顺序 = (
    "全部",
    "目标框内",
    "目标边界",
    "目标内部",
    "背景",
    "近距_<20m",
    "中距_20-40m",
    "远距_>=40m",
    "类别_Car",
    "类别_Pedestrian",
    "类别_Cyclist",
)
类别名称 = ("Car", "Pedestrian", "Cyclist")


@dataclass
class 在线指标:
    点数: int = 0
    绝对误差和_bin: float = 0.0
    命中1bin数: int = 0
    命中3bin数: int = 0
    排名和: float = 0.0
    排名前3数: int = 0
    间隔和: float = 0.0

    def 更新(
        self,
        掩码: np.ndarray,
        绝对误差: np.ndarray,
        排名: np.ndarray,
        间隔: np.ndarray,
    ) -> None:
        掩码 = np.asarray(掩码, dtype=bool)
        if not 掩码.any():
            return
        当前误差 = np.asarray(绝对误差, dtype=np.float64)[掩码]
        当前排名 = np.asarray(排名, dtype=np.float64)[掩码]
        当前间隔 = np.asarray(间隔, dtype=np.float64)[掩码]
        有效 = (
            np.isfinite(当前误差)
            & np.isfinite(当前排名)
            & np.isfinite(当前间隔)
        )
        if not 有效.any():
            return
        当前误差 = 当前误差[有效]
        当前排名 = 当前排名[有效]
        当前间隔 = 当前间隔[有效]
        self.点数 += int(当前误差.size)
        self.绝对误差和_bin += float(当前误差.sum())
        self.命中1bin数 += int((当前误差 <= 1.0).sum())
        self.命中3bin数 += int((当前误差 <= 3.0).sum())
        self.排名和 += float(当前排名.sum())
        self.排名前3数 += int((当前排名 <= 3.0).sum())
        self.间隔和 += float(当前间隔.sum())

    def 汇总(self) -> Dict[str, float]:
        return {
            "点数": self.点数,
            "平均绝对误差_bin": 安全除法(self.绝对误差和_bin, self.点数),
            "命中率_±1bin": 安全除法(self.命中1bin数, self.点数),
            "命中率_±3bin": 安全除法(self.命中3bin数, self.点数),
            "平均真值窗口排名": 安全除法(self.排名和, self.点数),
            "真值窗口Top3率": 安全除法(self.排名前3数, self.点数),
            "平均匹配间隔": 安全除法(self.间隔和, self.点数),
        }


def 安全除法(分子: float, 分母: float) -> float:
    return float(分子 / 分母) if 分母 else math.nan


def 解析整数列表(文本: str) -> List[int]:
    结果 = sorted({int(项目.strip()) for 项目 in 文本.split(",") if 项目.strip()})
    if not 结果 or any(项目 <= 0 for 项目 in 结果):
        raise ValueError("列表必须包含正整数")
    return 结果


def 构建点云成员映射(压缩包: zipfile.ZipFile) -> Dict[str, str]:
    return {
        Path(名称).stem: 名称
        for 名称 in 压缩包.namelist()
        if 名称.lower().endswith(".bin")
    }


def 从压缩包读取点云(
    压缩包: zipfile.ZipFile,
    样本号: str,
    成员映射: Mapping[str, str],
) -> np.ndarray:
    成员 = 成员映射.get(样本号)
    if 成员 is None:
        raise KeyError("Velodyne ZIP 中没有 {}.bin".format(样本号))
    数组 = np.frombuffer(压缩包.read(成员), dtype=np.float32)
    if 数组.size % 4:
        raise ValueError("{} 的 float32 数量不是 4 的倍数".format(成员))
    return 数组.reshape(-1, 4)


def 投影点云到网络输入(
    点云: np.ndarray,
    标定,
    原图大小: Sequence[float],
    目标大小: Sequence[int],
    顶部裁剪: int,
) -> Dict[str, np.ndarray]:
    """用与数据集相同的等比例仿射变换投影 P2/P3，并做像素 z-buffer。"""
    from lib.datasets.kitti.kitti_utils import affine_transform, get_affine_transform

    原宽, 原高 = float(原图大小[0]), float(原图大小[1])
    目标宽, 目标高 = int(目标大小[0]), int(目标大小[1])
    矩形点 = 标定.lidar_to_rect(点云[:, :3])
    前方 = 矩形点[:, 2] > 1.0e-6
    矩形点 = 矩形点[前方]
    if not len(矩形点):
        return {键: np.empty(0, dtype=np.float32) for 键 in ("x", "y", "z", "disp")}

    齐次 = np.concatenate(
        [矩形点.astype(np.float64), np.ones((len(矩形点), 1), dtype=np.float64)],
        axis=1,
    )
    左投影 = 齐次 @ 标定.P2.astype(np.float64).T
    右投影 = 齐次 @ 标定.P3.astype(np.float64).T
    左点 = 左投影[:, :2] / 左投影[:, 2:3]
    右点 = 右投影[:, :2] / 右投影[:, 2:3]

    变换, _ = get_affine_transform(
        np.array([原宽, 原高], dtype=np.float32) / 2.0,
        np.array([原宽, 原高], dtype=np.float32),
        0,
        np.array([目标宽, 目标高], dtype=np.float32),
        inv=1,
    )
    左网络 = np.stack([affine_transform(点, 变换) for 点 in 左点])
    右网络 = np.stack([affine_transform(点, 变换) for 点 in 右点])
    x = 左网络[:, 0]
    y = 左网络[:, 1] - float(顶部裁剪)
    视差 = 左网络[:, 0] - 右网络[:, 0]
    深度 = 矩形点[:, 2]
    裁剪高 = 目标高 - 顶部裁剪
    有效 = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(视差)
        & np.isfinite(深度)
        & (x >= 0.0)
        & (x < 目标宽)
        & (y >= 0.0)
        & (y < 裁剪高)
        & (视差 > 0.0)
        & (深度 > 0.0)
    )
    x, y, 深度, 视差 = x[有效], y[有效], 深度[有效], 视差[有效]
    if not len(x):
        return {键: np.empty(0, dtype=np.float32) for 键 in ("x", "y", "z", "disp")}

    像素x = np.floor(x).astype(np.int64)
    像素y = np.floor(y).astype(np.int64)
    像素键 = 像素y * 目标宽 + 像素x
    排序 = np.lexsort((深度, 像素键))
    排序键 = 像素键[排序]
    保留 = np.ones(len(排序), dtype=bool)
    保留[1:] = 排序键[1:] != 排序键[:-1]
    索引 = 排序[保留]
    return {
        "x": x[索引].astype(np.float32),
        "y": y[索引].astype(np.float32),
        "z": 深度[索引].astype(np.float32),
        "disp": 视差[索引].astype(np.float32),
    }


def 点区域掩码(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    targets: Mapping[str, object],
    batch_index: int,
    输入宽: int,
    输入高: int,
    边界宽度: float,
) -> Dict[str, np.ndarray]:
    """按训练目标中的变换后 2D 框给稀疏点分层。"""
    boxes_tensor = targets["boxes"][batch_index]
    labels_tensor = targets["labels"][batch_index]
    depth_tensor = targets["depth"][batch_index]
    boxes = boxes_tensor.detach().cpu().numpy()
    labels = labels_tensor.detach().cpu().numpy().astype(np.int64)
    depths = depth_tensor.detach().cpu().numpy().reshape(-1)
    valid_boxes = (depths > 0.0) & (boxes[:, 2] > 0.0) & (boxes[:, 3] > 0.0)
    boxes = boxes[valid_boxes]
    labels = labels[valid_boxes]
    object_depths = depths[valid_boxes]

    assigned = np.full(len(x), -1, dtype=np.int64)
    boundary = np.zeros(len(x), dtype=bool)
    best_area = np.full(len(x), np.inf, dtype=np.float64)
    for box, label, object_depth in zip(boxes, labels, object_depths):
        cx, cy, w, h = [float(item) for item in box]
        x1 = (cx - w / 2.0) * 输入宽
        x2 = (cx + w / 2.0) * 输入宽
        y1 = (cy - h / 2.0) * 输入高
        y2 = (cy + h / 2.0) * 输入高
        # 只把深度与目标中心相容的激光点归入目标，避免把框内远处背景
        # 错当成 Car/Pedestrian/Cyclist 的匹配证据。
        depth_tolerance = max(2.0, 0.15 * float(object_depth))
        inside = (
            (x >= x1)
            & (x <= x2)
            & (y >= y1)
            & (y <= y2)
            & (np.abs(z - float(object_depth)) <= depth_tolerance)
        )
        area = max((x2 - x1) * (y2 - y1), 1.0)
        update = inside & (area < best_area)
        if not update.any():
            continue
        edge_distance = np.minimum.reduce(
            [np.abs(x - x1), np.abs(x - x2), np.abs(y - y1), np.abs(y - y2)]
        )
        assigned[update] = int(label)
        boundary[update] = edge_distance[update] <= float(边界宽度)
        best_area[update] = area

    object_mask = assigned >= 0
    结果 = {
        "全部": np.ones(len(x), dtype=bool),
        "目标框内": object_mask,
        "目标边界": object_mask & boundary,
        "目标内部": object_mask & ~boundary,
        "背景": ~object_mask,
        "近距_<20m": z < 20.0,
        "中距_20-40m": (z >= 20.0) & (z < 40.0),
        "远距_>=40m": z >= 40.0,
    }
    for label, name in enumerate(类别名称):
        结果["类别_{}".format(name)] = assigned == label
    return 结果


def 分层抽样(
    投影: Dict[str, np.ndarray],
    区域: Dict[str, np.ndarray],
    每区最多点数: int,
    rng: np.random.RandomState,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    object_indices = np.flatnonzero(区域["目标框内"])
    background_indices = np.flatnonzero(区域["背景"])

    def 取样(indices: np.ndarray) -> np.ndarray:
        if len(indices) <= 每区最多点数:
            return indices
        return np.sort(rng.choice(indices, size=每区最多点数, replace=False))

    选择 = np.unique(np.concatenate([取样(object_indices), 取样(background_indices)]))
    if not len(选择):
        return 投影, 区域
    return (
        {键: 值[选择] for 键, 值 in 投影.items()},
        {键: 值[选择] for 键, 值 in 区域.items()},
    )


def 稀疏分组相关曲线(
    left,
    right,
    x: np.ndarray,
    y: np.ndarray,
    disp: np.ndarray,
    max_disp: int,
    num_groups: int,
):
    """在稀疏点处构建 [N,G,D] 分组相关，不生成整幅 G×D 代价体。"""
    import torch

    if left.ndim != 3 or right.shape != left.shape:
        raise ValueError("左右特征必须是相同形状 [C,H,W]")
    channels, height, width = left.shape
    if channels % num_groups:
        raise ValueError("特征通道 {} 不能被组数 {} 整除".format(channels, num_groups))
    scale_x = 1280.0 / float(width)
    scale_y = 288.0 / float(height)
    x_index = np.rint(x / scale_x).astype(np.int64)
    y_index = np.rint(y / scale_y).astype(np.int64)
    gt_index = np.rint(disp / scale_x).astype(np.int64)
    valid = (
        (x_index >= 0)
        & (x_index < width)
        & (y_index >= 0)
        & (y_index < height)
        & (gt_index >= 0)
        & (gt_index < max_disp)
        & (x_index >= gt_index)
    )
    if not valid.any():
        return None
    valid_indices = np.flatnonzero(valid)
    x_index = torch.as_tensor(x_index[valid], device=left.device, dtype=torch.long)
    y_index = torch.as_tensor(y_index[valid], device=left.device, dtype=torch.long)
    gt_index_tensor = torch.as_tensor(
        gt_index[valid], device=left.device, dtype=torch.long
    )
    count = int(x_index.numel())
    channels_per_group = channels // num_groups
    scores = left.new_full((count, num_groups, max_disp), -float("inf"))
    equivalence_max_error = 0.0
    left_vector = left[:, y_index, x_index].transpose(0, 1)
    for disparity in range(max_disp):
        disparity_valid = x_index >= disparity
        if not disparity_valid.any():
            continue
        selected_y = y_index[disparity_valid]
        selected_x = x_index[disparity_valid]
        right_vector = right[
            :, selected_y, selected_x - disparity
        ].transpose(0, 1)
        product = left_vector[disparity_valid] * right_vector
        grouped = product.reshape(-1, num_groups, channels_per_group).mean(dim=2)
        scores[disparity_valid, :, disparity] = grouped
        direct = product.mean(dim=1)
        difference = (grouped.mean(dim=1) - direct).abs().max().item()
        equivalence_max_error = max(equivalence_max_error, float(difference))
    return scores, gt_index_tensor, valid_indices, equivalence_max_error


def 曲线指标(scores, gt_index, 真值窗口半径: int = 1):
    """返回每点误差、真值窗口排名和相对最强错误视差的匹配间隔。"""
    import torch

    if scores.ndim != 2:
        raise ValueError("scores 必须是 [N,D]")
    count, disparity_count = scores.shape
    prediction = scores.argmax(dim=1)
    error = (prediction - gt_index).abs().float()
    disparity_axis = torch.arange(disparity_count, device=scores.device).view(1, -1)
    true_window = (disparity_axis - gt_index.view(-1, 1)).abs() <= 真值窗口半径
    true_score = scores.masked_fill(~true_window, -float("inf")).max(dim=1).values
    wrong_score = scores.masked_fill(true_window, -float("inf")).max(dim=1).values
    rank = 1 + (scores > true_score.view(-1, 1)).sum(dim=1)
    margin = torch.where(
        torch.isfinite(wrong_score),
        true_score - wrong_score,
        torch.zeros_like(true_score),
    )
    return error, rank.float(), margin


def 三种方法指标(group_scores, gt_index):
    """当前均匀平均、分组最大和逐点最优组 Oracle。"""
    import torch

    uniform_scores = group_scores.mean(dim=1)
    max_scores = group_scores.max(dim=1).values
    uniform = 曲线指标(uniform_scores, gt_index)
    group_max = 曲线指标(max_scores, gt_index)

    group_metrics = [曲线指标(group_scores[:, group], gt_index) for group in range(group_scores.shape[1])]
    group_errors = torch.stack([item[0] for item in group_metrics], dim=1)
    group_ranks = torch.stack([item[1] for item in group_metrics], dim=1)
    group_margins = torch.stack([item[2] for item in group_metrics], dim=1)
    best_group = group_errors.argmin(dim=1)
    row = torch.arange(group_errors.shape[0], device=group_errors.device)
    oracle = (
        group_errors[row, best_group],
        group_ranks[row, best_group],
        group_margins[row, best_group],
    )
    return {
        "当前均匀平均": uniform,
        "分组最大聚合": group_max,
        "逐点最优分组_Oracle": oracle,
    }


def 找到深度预测器(model):
    if hasattr(model, "depth_predictor"):
        return model.depth_predictor
    if hasattr(model, "base_model") and hasattr(model.base_model, "depth_predictor"):
        return model.base_model.depth_predictor
    raise AttributeError("模型中找不到 depth_predictor")


def 生成判断(
    汇总: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    实际样本数: int,
) -> Dict[str, object]:
    """预注册止损：正式诊断要求目标点充分且 Oracle 同时改善命中与误差。"""
    候选 = []
    for scale_key, groups_data in 汇总.items():
        for group_key, methods in groups_data.items():
            baseline = methods["当前均匀平均"]["目标框内"]
            oracle = methods["逐点最优分组_Oracle"]["目标框内"]
            car_baseline = methods["当前均匀平均"]["类别_Car"]
            car_oracle = methods["逐点最优分组_Oracle"]["类别_Car"]
            points = int(baseline["点数"])
            car_points = int(car_baseline["点数"])
            hit_gain = (
                float(oracle["命中率_±1bin"]) - float(baseline["命中率_±1bin"])
                if points else math.nan
            )
            error_reduction = (
                1.0 - float(oracle["平均绝对误差_bin"]) / max(
                    float(baseline["平均绝对误差_bin"]), 1.0e-9
                )
                if points else math.nan
            )
            car_hit_gain = (
                float(car_oracle["命中率_±1bin"])
                - float(car_baseline["命中率_±1bin"])
                if car_points else math.nan
            )
            car_error_reduction = (
                1.0 - float(car_oracle["平均绝对误差_bin"]) / max(
                    float(car_baseline["平均绝对误差_bin"]), 1.0e-9
                )
                if car_points else math.nan
            )
            候选.append(
                {
                    "尺度": scale_key,
                    "组数": group_key,
                    "目标点数": points,
                    "Oracle命中提升": hit_gain,
                    "Oracle误差相对下降": error_reduction,
                    "Car点数": car_points,
                    "Car_Oracle命中提升": car_hit_gain,
                    "Car_Oracle误差相对下降": car_error_reduction,
                }
            )
    best = max(
        候选,
        key=lambda item: (
            -math.inf
            if not np.isfinite(item["Oracle命中提升"])
            else (
                item["Oracle命中提升"]
                + item["Oracle误差相对下降"]
                + (
                    item["Car_Oracle命中提升"]
                    if np.isfinite(item["Car_Oracle命中提升"])
                    else 0.0
                )
                + (
                    item["Car_Oracle误差相对下降"]
                    if np.isfinite(item["Car_Oracle误差相对下降"])
                    else 0.0
                )
            )
        ),
    )
    evidence_ready = (
        实际样本数 >= 100
        and best["目标点数"] >= 5000
        and best["Car点数"] >= 3000
    )
    information_pass = (
        np.isfinite(best["Oracle命中提升"])
        and best["Oracle命中提升"] >= 0.05
        and best["Oracle误差相对下降"] >= 0.10
        and np.isfinite(best["Car_Oracle命中提升"])
        and best["Car_Oracle命中提升"] >= 0.03
        and best["Car_Oracle误差相对下降"] >= 0.05
    )
    if not evidence_ready:
        status = "证据不足"
        explanation = "样本数或目标框内稀疏点不足，不能据此修改主模型。"
    elif information_pass:
        status = "通过进入轻量门控短程配对"
        explanation = (
            "分组相关保留了均匀平均丢失的显著信息；下一步仅实现零初始化、"
            "压回原D通道的门控，并先做同起点短程配对。"
        )
    else:
        status = "否决"
        explanation = "分组 Oracle 上限没有同时达到命中率和误差门槛，停止 V12。"
    return {"状态": status, "解释": explanation, "最佳候选": best, "全部候选": 候选}


def 写出结果(
    output_dir: Path,
    汇总,
    判断,
    元数据,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "诊断明细.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {"元数据": 元数据, "判断": 判断, "汇总": 汇总},
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )
    with (output_dir / "诊断汇总.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "尺度", "组数", "方法", "分层", "点数", "平均绝对误差_bin",
                "命中率_±1bin", "命中率_±3bin", "平均真值窗口排名",
                "真值窗口Top3率", "平均匹配间隔",
            ]
        )
        for scale_key, groups_data in 汇总.items():
            for group_key, methods in groups_data.items():
                for method, regions in methods.items():
                    for region, values in regions.items():
                        writer.writerow(
                            [scale_key, group_key, method, region]
                            + [
                                values[key]
                                for key in (
                                    "点数", "平均绝对误差_bin", "命中率_±1bin",
                                    "命中率_±3bin", "平均真值窗口排名",
                                    "真值窗口Top3率", "平均匹配间隔",
                                )
                            ]
                        )

    best = 判断["最佳候选"]
    scale_data = 汇总[best["尺度"]][best["组数"]]
    lines = [
        "# StereoDETR V12-G0 分组相关信息诊断报告",
        "",
        "## 结论先行",
        "",
        "- 自动结论：**{}**。{}".format(判断["状态"], 判断["解释"]),
        "- 最佳诊断组合：{}，{}。".format(best["尺度"], best["组数"]),
        "- 目标框内 Oracle 的 ±1 bin 命中率提升：**{:.2f} 个百分点**。".format(
            100.0 * float(best["Oracle命中提升"])
        ),
        "- 目标框内 Oracle 的平均 bin 误差相对下降：**{:.2f}%**。".format(
            100.0 * float(best["Oracle误差相对下降"])
        ),
        "- Car Oracle 的 ±1 bin 命中率提升/误差相对下降：**{:.2f} 个百分点 / {:.2f}%**。".format(
            100.0 * float(best["Car_Oracle命中提升"]),
            100.0 * float(best["Car_Oracle误差相对下降"]),
        ),
        "- Oracle 使用激光真值逐点选择通道组，禁止当作模型精度或论文结果。",
        "- 即便通过，本报告也只授权下一步 5 轮同起点配对，不授权直接跑 195 轮。",
        "",
        "## 最佳组合分层结果",
        "",
        "| 分层 | 方法 | 点数 | MAE(bin) | ±1bin | ±3bin | 真值Top3 | 匹配间隔 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for region in 分组顺序:
        for method in 方法顺序:
            values = scale_data[method][region]
            lines.append(
                "| {} | {} | {} | {:.4f} | {:.2f}% | {:.2f}% | {:.2f}% | {:.5f} |".format(
                    region,
                    method,
                    int(values["点数"]),
                    float(values["平均绝对误差_bin"]),
                    100.0 * float(values["命中率_±1bin"]),
                    100.0 * float(values["命中率_±3bin"]),
                    100.0 * float(values["真值窗口Top3率"]),
                    float(values["平均匹配间隔"]),
                )
            )
    lines.extend(
        [
            "",
            "## 证据门槛",
            "",
            "正式判断要求：至少 100 个样本、最佳组合至少 5000 个目标框内点和",
            "3000 个 Car 点。目标点还必须同时落入变换后 2D 框，且与目标中心",
            "深度相差不超过 max(2m, 15%)，以排除框内背景污染。",
            "且 Oracle 的 ±1 bin 命中率提升不少于 5 个百分点、平均 bin 误差",
            "相对下降不少于 10%；Car 还需分别达到 3 个百分点和 5%。",
            "四项必须同时满足。",
            "",
            "## 基线等价性审计",
            "",
            "- 各组均匀平均与当前全通道平均的最大绝对差：`{:.8g}`。".format(
                float(元数据["均匀平均等价最大误差"])
            ),
            "- 诊断样本：{}；成功：{}；失败：{}；总耗时：{:.2f} 分钟。".format(
                元数据["计划样本数"], 元数据["成功样本数"],
                len(元数据["失败样本"]), float(元数据["总耗时秒"]) / 60.0,
            ),
            "",
            "完整数值见 `诊断汇总.csv` 和 `诊断明细.json`。",
        ]
    )
    (output_dir / "诊断报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def 自检() -> None:
    import torch

    torch.manual_seed(7)
    left = torch.randn(8, 3, 9)
    right = torch.randn(8, 3, 9)
    x = np.array([4.0, 6.0], dtype=np.float32) * 1280.0 / 9.0
    y = np.array([1.0, 2.0], dtype=np.float32) * 288.0 / 3.0
    disp = np.array([1.0, 2.0], dtype=np.float32) * 1280.0 / 9.0
    result = 稀疏分组相关曲线(left, right, x, y, disp, 4, 4)
    assert result is not None
    scores, gt, _, equivalence_error = result
    assert equivalence_error < 1.0e-6
    uniform = scores.mean(dim=1)
    for row in range(2):
        xi = int(round(float(x[row]) / (1280.0 / 9.0)))
        yi = int(round(float(y[row]) / (288.0 / 3.0)))
        for d in range(min(4, xi + 1)):
            direct = (left[:, yi, xi] * right[:, yi, xi - d]).mean()
            assert torch.allclose(uniform[row, d], direct, atol=1.0e-6)

    synthetic = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 2.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    synthetic_gt = torch.tensor([1, 2])
    metrics = 三种方法指标(synthetic, synthetic_gt)
    assert torch.all(metrics["逐点最优分组_Oracle"][0] == 0)

    mock = {
        "尺度_s4": {
            "组数_8": {
                method: {
                    region: {
                        "点数": 6000,
                        "平均绝对误差_bin": 2.0 if method == "当前均匀平均" else 1.0,
                        "命中率_±1bin": 0.50 if method == "当前均匀平均" else 0.60,
                    }
                    for region in 分组顺序
                }
                for method in 方法顺序
            }
        }
    }
    assert 生成判断(mock, 100)["状态"] == "通过进入轻量门控短程配对"
    assert 生成判断(mock, 20)["状态"] == "证据不足"
    print("V12-G0自检通过：分组相关、均匀等价、Oracle指标和止损门槛正常")


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="versions/V09_V08O加点式三维质量排序/config.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/最终组合_V09/V09_V08O加点式三维质量排序/"
            "qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
        ),
    )
    parser.add_argument(
        "--velodyne_zip",
        type=Path,
        default=Path("/autodl-pub/data/KITTI_Object/raw/data_object_velodyne.zip"),
    )
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--groups", default="4,8,16")
    parser.add_argument("--scales", default="4,8,16")
    parser.add_argument("--points_per_region", type=int, default=384)
    parser.add_argument("--box_edge_width", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/第二创新点_V12分组相关诊断/G0_100样本"),
    )
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def 主程序(args: argparse.Namespace) -> int:
    if args.self_test:
        自检()
        return 0

    import torch
    from torch.utils.data import DataLoader, Subset
    import tqdm

    from lib.helpers.config_helper import load_config
    from lib.helpers.qecr_dataloader_helper import build_qecr_dataloader
    from lib.helpers.qecr_model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("V12-G0完整诊断需要 CUDA 运行 V09 冻结模型")
    if args.num_samples <= 0 or args.points_per_region <= 0:
        raise ValueError("样本数和每区点数必须为正")
    groups = 解析整数列表(args.groups)
    scales = 解析整数列表(args.scales)
    scale_to_level = {4: 0, 8: 1, 16: 2}
    scale_to_max_disp = {4: 24, 8: 24, 16: 12}
    if not set(scales).issubset(scale_to_level):
        raise ValueError("--scales 只支持 4,8,16")
    if not args.velodyne_zip.is_file():
        raise FileNotFoundError(args.velodyne_zip)

    start = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    _, full_loader = build_qecr_dataloader(cfg["dataset"], workers=args.workers)
    dataset = full_loader.dataset
    sample_count = min(args.num_samples, len(dataset))
    rng = np.random.RandomState(args.seed)
    selected = np.sort(rng.choice(len(dataset), size=sample_count, replace=False))
    loader = DataLoader(
        Subset(dataset, selected.tolist()),
        batch_size=int(cfg["dataset"]["batch_size"]),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
    )

    model, _ = build_model(cfg["model"])
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    logger = logging.getLogger("StereoDETR_V12G0")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    load_checkpoint(model, None, str(checkpoint), device, logger=logger)

    captured: Dict[str, Tuple[torch.Tensor, ...]] = {}

    def capture_input(_module, module_inputs):
        captured["features"] = tuple(module_inputs[0][:3])

    hook = 找到深度预测器(model).register_forward_pre_hook(capture_input)
    accumulators: MutableMapping[
        Tuple[str, str, str, str], 在线指标
    ] = defaultdict(在线指标)
    failures: List[Dict[str, str]] = []
    completed = 0
    equivalence_max_error = 0.0
    total_sparse_points = 0
    progress = tqdm.tqdm(total=sample_count, desc="V12-G0")

    try:
        with zipfile.ZipFile(args.velodyne_zip, "r") as archive:
            member_map = 构建点云成员映射(archive)
            with torch.no_grad():
                for inputs, calibs, targets, info in loader:
                    inputs = inputs.to(device)
                    calibs_gpu = calibs.to(device)
                    img_sizes = info["img_size_croped"].to(device)
                    img_sizes_ori = info["img_size_original"].to(device)
                    img_sizes_upper = info["upper"].to(device)
                    captured.clear()
                    _ = model(
                        inputs,
                        calibs_gpu,
                        targets,
                        img_sizes,
                        img_sizes_ori,
                        img_sizes_upper,
                        dn_args=0,
                    )
                    features = captured.get("features")
                    if features is None:
                        raise RuntimeError("前向钩子没有捕获 depth_predictor 输入")
                    batch_size = inputs.shape[0]
                    for batch_index in range(batch_size):
                        sample_id = str(info["img_id"][batch_index])
                        try:
                            original_size = (
                                info["img_size_original"][batch_index]
                                .detach().cpu().numpy()
                            )
                            calibration = dataset.get_calib(sample_id)
                            cloud = 从压缩包读取点云(
                                archive, sample_id, member_map
                            )
                            projection = 投影点云到网络输入(
                                cloud,
                                calibration,
                                original_size,
                                cfg["dataset"]["resolution_ori"],
                                int(cfg["dataset"]["crop_top"]),
                            )
                            if not len(projection["x"]):
                                raise ValueError("没有有效激光投影点")
                            input_width = int(cfg["dataset"]["resolution_ori"][0])
                            input_height = int(
                                cfg["dataset"]["resolution_ori"][1]
                                - cfg["dataset"]["crop_top"]
                            )
                            regions = 点区域掩码(
                                projection["x"], projection["y"], projection["z"],
                                targets, batch_index, input_width, input_height,
                                args.box_edge_width,
                            )
                            projection, regions = 分层抽样(
                                projection, regions, args.points_per_region, rng
                            )
                            total_sparse_points += len(projection["x"])

                            for scale in scales:
                                level = scale_to_level[scale]
                                stereo_feature = features[level]
                                left = stereo_feature[batch_index]
                                right = stereo_feature[batch_index + batch_size]
                                for num_groups in groups:
                                    if left.shape[0] % num_groups:
                                        continue
                                    result = 稀疏分组相关曲线(
                                        left, right,
                                        projection["x"], projection["y"],
                                        projection["disp"],
                                        scale_to_max_disp[scale], num_groups,
                                    )
                                    if result is None:
                                        continue
                                    (
                                        group_scores,
                                        gt_index,
                                        valid_indices,
                                        equivalence_error,
                                    ) = result
                                    # 均匀分组平均必须数值恢复当前全通道平均。
                                    equivalence_max_error = max(
                                        equivalence_max_error,
                                        float(equivalence_error),
                                    )
                                    method_metrics = 三种方法指标(group_scores, gt_index)
                                    scale_key = "尺度_s{}".format(scale)
                                    group_key = "组数_{}".format(num_groups)
                                    for method, tensors in method_metrics.items():
                                        error, rank, margin = [
                                            tensor.detach().cpu().numpy() for tensor in tensors
                                        ]
                                        for region_name in 分组顺序:
                                            mask = regions[region_name][valid_indices]
                                            accumulators[
                                                (scale_key, group_key, method, region_name)
                                            ].更新(mask, error, rank, margin)
                            completed += 1
                        except Exception as error:
                            failures.append({"样本": sample_id, "错误": repr(error)})
                            print(
                                "[警告] {} 诊断失败：{}".format(sample_id, error),
                                file=sys.stderr,
                                flush=True,
                            )
                        progress.update(1)
    finally:
        hook.remove()
        progress.close()

    if completed == 0:
        raise RuntimeError("所有样本均失败")
    summary: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {}
    for scale in scales:
        scale_key = "尺度_s{}".format(scale)
        summary[scale_key] = {}
        for num_groups in groups:
            group_key = "组数_{}".format(num_groups)
            summary[scale_key][group_key] = {}
            for method in 方法顺序:
                summary[scale_key][group_key][method] = {
                    region: accumulators[
                        (scale_key, group_key, method, region)
                    ].汇总()
                    for region in 分组顺序
                }

    decision = 生成判断(summary, completed)
    metadata = {
        "配置": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint),
        "velodyne_zip": str(args.velodyne_zip),
        "计划样本数": sample_count,
        "成功样本数": completed,
        "失败样本": failures,
        "抽样种子": args.seed,
        "组数": groups,
        "尺度": scales,
        "每区最多点数": args.points_per_region,
        "累计稀疏点": total_sparse_points,
        "均匀平均等价最大误差": equivalence_max_error,
        "总耗时秒": time.time() - start,
    }
    写出结果(args.output_dir, summary, decision, metadata)
    print("\nV12-G0诊断完成")
    print("报告：{}".format(args.output_dir / "诊断报告.md"))
    print("自动结论：{}".format(decision["状态"]))
    print("最佳候选：{}".format(decision["最佳候选"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(主程序(解析参数()))
    except KeyboardInterrupt:
        print("\n用户中止 V12-G0", file=sys.stderr)
        raise SystemExit(130)
