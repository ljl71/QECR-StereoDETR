#!/usr/bin/env python3
"""G4：冻结 V09，诊断 s4 相关体是否值得增加选择性高/低频细化。

本脚本不训练、不修改 checkpoint，也不把 Velodyne 输入检测器。它只比较：

1. 当前 s4 原始相关曲线（高频/小感受野代理）；
2. 相关体 3x3 局部平均后的曲线（低频/大感受野代理）；
3. 逐点选择误差更小分支的 Oracle 信息上限；
4. 仅用校准样本拟合、在独立留出样本评估的左目上下文边缘阈值门控。

Oracle 使用真值，只能决定是否值得开发，不能当作模型结果。只有 Oracle 和
留出集代理门控同时满足预注册门槛，才允许实现正式 V13 适配器。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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


def 加载共享诊断函数():
    路径 = 项目根目录 / "tools" / "诊断分组相关信息_V12G0.py"
    规格 = importlib.util.spec_from_file_location("qecr_v12g0_shared", str(路径))
    if 规格 is None or 规格.loader is None:
        raise ImportError("无法加载 {}".format(路径))
    模块 = importlib.util.module_from_spec(规格)
    sys.modules[规格.name] = 模块
    规格.loader.exec_module(模块)
    return 模块


共享 = 加载共享诊断函数()
构建点云成员映射 = 共享.构建点云成员映射
从压缩包读取点云 = 共享.从压缩包读取点云
投影点云到网络输入 = 共享.投影点云到网络输入
点区域掩码 = 共享.点区域掩码
分层抽样 = 共享.分层抽样
找到深度预测器 = 共享.找到深度预测器


类别名称 = ("Car", "Pedestrian", "Cyclist")
区域顺序 = (
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
方法顺序 = ("当前原始相关", "固定局部平滑", "双分支_Oracle", "上下文阈值门控")


@dataclass
class 误差指标:
    点数: int = 0
    绝对误差和: float = 0.0
    命中1数: int = 0
    命中3数: int = 0

    def 更新(self, 掩码: np.ndarray, 误差: np.ndarray) -> None:
        掩码 = np.asarray(掩码, dtype=bool)
        当前 = np.asarray(误差, dtype=np.float64)[掩码]
        当前 = 当前[np.isfinite(当前)]
        if not 当前.size:
            return
        self.点数 += int(当前.size)
        self.绝对误差和 += float(当前.sum())
        self.命中1数 += int((当前 <= 1.0).sum())
        self.命中3数 += int((当前 <= 3.0).sum())

    def 汇总(self) -> Dict[str, float]:
        if not self.点数:
            return {
                "点数": 0,
                "平均绝对误差_bin": math.nan,
                "命中率_±1bin": math.nan,
                "命中率_±3bin": math.nan,
            }
        return {
            "点数": self.点数,
            "平均绝对误差_bin": self.绝对误差和 / self.点数,
            "命中率_±1bin": self.命中1数 / self.点数,
            "命中率_±3bin": self.命中3数 / self.点数,
        }


def 相关曲线误差(曲线, 真值索引):
    预测 = 曲线.argmax(dim=1)
    return (预测 - 真值索引).abs().float()


def 提取稀疏双频证据(
    原始相关体,
    平滑相关体,
    左特征,
    x: np.ndarray,
    y: np.ndarray,
    disp: np.ndarray,
):
    """返回稀疏点的两路误差、上下文边缘强度和原始点索引。"""
    import torch

    if 原始相关体.ndim != 3 or 平滑相关体.shape != 原始相关体.shape:
        raise ValueError("相关体必须是相同形状 [D,H,W]")
    if 左特征.ndim != 3:
        raise ValueError("左特征必须是 [C,H,W]")
    disparity_count, height, width = 原始相关体.shape
    scale_x = 1280.0 / float(width)
    scale_y = 288.0 / float(height)
    x_index = np.rint(x / scale_x).astype(np.int64)
    y_index = np.rint(y / scale_y).astype(np.int64)
    gt_index = np.rint(disp / scale_x).astype(np.int64)

    # 平滑分支采用 3x3，因此排除一圈边界。x>=D 可避免左边无效视差的
    # 零填充成为“平滑更好”的伪证据。
    valid = (
        (x_index >= disparity_count)
        & (x_index < width - 1)
        & (y_index >= 1)
        & (y_index < height - 1)
        & (gt_index >= 0)
        & (gt_index < disparity_count)
    )
    if not valid.any():
        return None

    原始索引 = np.flatnonzero(valid)
    xi = torch.as_tensor(x_index[valid], device=左特征.device, dtype=torch.long)
    yi = torch.as_tensor(y_index[valid], device=左特征.device, dtype=torch.long)
    gt = torch.as_tensor(gt_index[valid], device=左特征.device, dtype=torch.long)
    raw_curve = 原始相关体[:, yi, xi].transpose(0, 1)
    smooth_curve = 平滑相关体[:, yi, xi].transpose(0, 1)
    raw_error = 相关曲线误差(raw_curve, gt)
    smooth_error = 相关曲线误差(smooth_curve, gt)

    center = 左特征[:, yi, xi].transpose(0, 1)
    neighbors = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbors.append(
            左特征[:, yi + dy, xi + dx].transpose(0, 1)
        )
    gradient = torch.stack(
        [(item - center).abs().mean(dim=1) for item in neighbors], dim=1
    ).mean(dim=1)
    energy = center.abs().mean(dim=1).clamp_min(1.0e-6)
    edge = gradient / energy

    return {
        "原始点索引": 原始索引,
        "原始误差": raw_error.detach().cpu().numpy(),
        "平滑误差": smooth_error.detach().cpu().numpy(),
        "边缘强度": edge.detach().cpu().numpy(),
    }


def 拟合语义阈值(
    边缘强度: np.ndarray,
    原始误差: np.ndarray,
    平滑误差: np.ndarray,
    目标掩码: np.ndarray,
) -> Dict[str, float]:
    """高边缘固定选原始分支，低边缘选平滑分支；只搜索阈值。"""
    edge = np.asarray(边缘强度, dtype=np.float64)
    raw = np.asarray(原始误差, dtype=np.float64)
    smooth = np.asarray(平滑误差, dtype=np.float64)
    mask = np.asarray(目标掩码, dtype=bool)
    valid = mask & np.isfinite(edge) & np.isfinite(raw) & np.isfinite(smooth)
    if valid.sum() < 100:
        raise ValueError("校准集目标点不足 100，无法稳定拟合阈值")
    values = edge[valid]
    quantiles = np.linspace(0.02, 0.98, 97)
    candidates = np.unique(np.quantile(values, quantiles))
    best = None
    for threshold in candidates:
        choose_raw = edge >= threshold
        error = np.where(choose_raw, raw, smooth)[valid]
        objective = float(error.mean())
        hit1 = float((error <= 1.0).mean())
        item = (objective, -hit1, float(threshold))
        if best is None or item < best:
            best = item
    assert best is not None
    return {
        "阈值": best[2],
        "校准目标平均误差": best[0],
        "校准目标命中率_±1bin": -best[1],
    }


def 合并记录(记录列表: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not 记录列表:
        raise ValueError("没有可合并的记录")
    result: Dict[str, object] = {}
    for key in ("边缘强度", "原始误差", "平滑误差"):
        result[key] = np.concatenate([np.asarray(item[key]) for item in 记录列表])
    result["区域"] = {
        region: np.concatenate(
            [np.asarray(item["区域"][region], dtype=bool) for item in 记录列表]
        )
        for region in 区域顺序
    }
    return result


def 计算方法汇总(
    records: Mapping[str, object],
    threshold: float,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, float]]:
    raw = np.asarray(records["原始误差"], dtype=np.float64)
    smooth = np.asarray(records["平滑误差"], dtype=np.float64)
    edge = np.asarray(records["边缘强度"], dtype=np.float64)
    regions = records["区域"]
    oracle = np.minimum(raw, smooth)
    choose_raw = edge >= float(threshold)
    proxy = np.where(choose_raw, raw, smooth)
    method_errors = {
        "当前原始相关": raw,
        "固定局部平滑": smooth,
        "双分支_Oracle": oracle,
        "上下文阈值门控": proxy,
    }
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in 方法顺序:
        summary[method] = {}
        for region in 区域顺序:
            metric = 误差指标()
            metric.更新(np.asarray(regions[region], dtype=bool), method_errors[method])
            summary[method][region] = metric.汇总()

    def 选择比例(region: str) -> float:
        mask = np.asarray(regions[region], dtype=bool)
        return float(choose_raw[mask].mean()) if mask.any() else math.nan

    shares = {
        "目标框内选择原始比例": 选择比例("目标框内"),
        "目标边界选择原始比例": 选择比例("目标边界"),
        "目标内部选择原始比例": 选择比例("目标内部"),
        "背景选择原始比例": 选择比例("背景"),
    }
    return summary, shares


def 安全差(a: float, b: float) -> float:
    return float(a) - float(b) if math.isfinite(float(a)) and math.isfinite(float(b)) else math.nan


def 方法变化(
    summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    method: str,
    region: str,
) -> Dict[str, float]:
    base = summary["当前原始相关"][region]
    candidate = summary[method][region]
    base_error = float(base["平均绝对误差_bin"])
    candidate_error = float(candidate["平均绝对误差_bin"])
    reduction = (
        1.0 - candidate_error / max(base_error, 1.0e-9)
        if math.isfinite(base_error) and math.isfinite(candidate_error)
        else math.nan
    )
    return {
        "点数": int(base["点数"]),
        "命中1变化": 安全差(candidate["命中率_±1bin"], base["命中率_±1bin"]),
        "误差相对下降": reduction,
    }


def 生成判断(
    summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    shares: Mapping[str, float],
    success_samples: int,
) -> Dict[str, object]:
    oracle_object = 方法变化(summary, "双分支_Oracle", "目标框内")
    oracle_car = 方法变化(summary, "双分支_Oracle", "类别_Car")
    proxy_object = 方法变化(summary, "上下文阈值门控", "目标框内")
    proxy_car = 方法变化(summary, "上下文阈值门控", "类别_Car")
    boundary_gap = 安全差(
        shares["目标边界选择原始比例"],
        shares["目标内部选择原始比例"],
    )

    enough = (
        success_samples >= 80
        and oracle_object["点数"] >= 500
        and oracle_car["点数"] >= 300
    )
    oracle_pass = all(
        item["命中1变化"] >= 0.03 and item["误差相对下降"] >= 0.05
        for item in (oracle_object, oracle_car)
    )
    proxy_pass = all(
        item["命中1变化"] >= 0.005 and item["误差相对下降"] >= 0.01
        for item in (proxy_object, proxy_car)
    )

    guards = []
    for region in ("类别_Pedestrian", "类别_Cyclist"):
        item = 方法变化(summary, "上下文阈值门控", region)
        # 稀疏点太少时不宣称通过，避免小类别保护被空样本绕过。
        guards.append(
            item["点数"] >= 50
            and item["命中1变化"] >= -0.005
            and item["误差相对下降"] >= -0.02
        )
    semantic_pass = math.isfinite(boundary_gap) and boundary_gap >= 0.10

    if not enough:
        status = "证据不足，停止正式改码"
        explanation = "成功样本或目标/Car稀疏点不足预注册下限。"
    elif not oracle_pass:
        status = "否决，停止V13"
        explanation = "高低频双分支本身没有提供足够的目标与Car信息上限。"
    elif not proxy_pass:
        status = "否决，停止V13"
        explanation = "Oracle虽存在，但低成本上下文门控无法在独立留出集兑现。"
    elif not all(guards):
        status = "否决，停止V13"
        explanation = "Pedestrian/Cyclist留出集保护线未通过。"
    elif not semantic_pass:
        status = "否决，停止V13"
        explanation = "边界与内部的分支选择差异不足，频率选择解释不成立。"
    else:
        status = "通过，允许实现V13配对短训"
        explanation = "Oracle、留出门控、小类别保护和频率语义均达到预注册门槛。"

    return {
        "状态": status,
        "解释": explanation,
        "成功样本门槛通过": enough,
        "Oracle门槛通过": oracle_pass,
        "代理门控门槛通过": proxy_pass,
        "小类别保护通过": all(guards),
        "频率语义通过": semantic_pass,
        "边界减内部_选择原始比例": boundary_gap,
        "Oracle目标变化": oracle_object,
        "OracleCar变化": oracle_car,
        "门控目标变化": proxy_object,
        "门控Car变化": proxy_car,
    }


def 格式百分比(value: float) -> str:
    return "—" if not math.isfinite(float(value)) else "{:+.2f}".format(100.0 * float(value))


def 格式小数(value: float) -> str:
    return "—" if not math.isfinite(float(value)) else "{:.4f}".format(float(value))


def 写出结果(
    output_dir: Path,
    summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    shares: Mapping[str, float],
    calibration: Mapping[str, float],
    decision: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "元数据": dict(metadata),
        "校准阈值": dict(calibration),
        "留出集汇总": summary,
        "分支选择比例": dict(shares),
        "自动判断": dict(decision),
    }
    (output_dir / "诊断明细.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    with (output_dir / "诊断汇总.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["方法", "区域", "点数", "平均绝对误差_bin", "命中率_±1bin", "命中率_±3bin"])
        for method in 方法顺序:
            for region in 区域顺序:
                item = summary[method][region]
                writer.writerow([
                    method,
                    region,
                    item["点数"],
                    item["平均绝对误差_bin"],
                    item["命中率_±1bin"],
                    item["命中率_±3bin"],
                ])

    lines = [
        "# StereoDETR 选择性频率融合 G4 诊断报告",
        "",
        "## 结论先行",
        "",
        "- 自动结论：**{}**。{}".format(decision["状态"], decision["解释"]),
        "- 本报告只比较冻结特征上的稀疏视差证据，不是3D检测实验结果。",
        "- Oracle使用Velodyne真值逐点选分支，不能写成模型精度。",
        "- 阈值只在校准样本拟合，下表全部来自样本级隔离的留出集。",
        "",
        "## 数据与阈值",
        "",
        "- 成功样本：{}（校准 {}，留出 {}）".format(
            metadata["成功样本数"], metadata["校准成功样本数"], metadata["留出成功样本数"]
        ),
        "- 校准得到的上下文边缘阈值：`{}`".format(格式小数(calibration["阈值"])),
        "- 留出集累计稀疏点：{}".format(metadata["留出稀疏点数"]),
        "",
        "## 留出集目标区域",
        "",
        "| 方法 | 目标点 | MAE(bin) | ±1bin | ±3bin |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in 方法顺序:
        item = summary[method]["目标框内"]
        lines.append(
            "| {} | {} | {} | {:.2f}% | {:.2f}% |".format(
                method,
                item["点数"],
                格式小数(item["平均绝对误差_bin"]),
                100.0 * item["命中率_±1bin"] if math.isfinite(item["命中率_±1bin"]) else math.nan,
                100.0 * item["命中率_±3bin"] if math.isfinite(item["命中率_±3bin"]) else math.nan,
            )
        )

    lines.extend([
        "",
        "## 留出集相对当前原始相关的变化",
        "",
        "| 方法/区域 | ±1bin变化(百分点) | MAE相对下降 |",
        "|---|---:|---:|",
    ])
    for method in ("固定局部平滑", "双分支_Oracle", "上下文阈值门控"):
        for region in ("目标框内", "类别_Car", "类别_Pedestrian", "类别_Cyclist"):
            item = 方法变化(summary, method, region)
            lines.append(
                "| {} / {} | {} | {}% |".format(
                    method,
                    region,
                    格式百分比(item["命中1变化"]),
                    格式百分比(item["误差相对下降"]),
                )
            )

    lines.extend([
        "",
        "## 分支选择语义",
        "",
        "| 区域 | 选择原始高频分支比例 |",
        "|---|---:|",
    ])
    for key, value in shares.items():
        lines.append("| {} | {:.2f}% |".format(key, 100.0 * value))
    lines.extend([
        "",
        "- 边界减内部：`{}` 个百分点；预注册要求至少 `+10.00`。".format(
            格式百分比(decision["边界减内部_选择原始比例"])
        ),
        "",
        "## 预注册门槛审计",
        "",
        "| 门槛 | 是否通过 |",
        "|---|---|",
        "| 样本与稀疏点充分 | {} |".format(decision["成功样本门槛通过"]),
        "| Oracle目标/Car收益 | {} |".format(decision["Oracle门槛通过"]),
        "| 留出集低成本门控收益 | {} |".format(decision["代理门控门槛通过"]),
        "| Pedestrian/Cyclist保护 | {} |".format(decision["小类别保护通过"]),
        "| 边界/内部频率语义 | {} |".format(decision["频率语义通过"]),
        "",
        "只有最终状态为“通过，允许实现V13配对短训”时，才能修改正式模型。",
    ])
    (output_dir / "诊断报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def 自检() -> None:
    import torch

    raw_volume = torch.zeros(4, 5, 8)
    smooth_volume = torch.zeros_like(raw_volume)
    raw_volume[1, 2, 5] = 3.0
    smooth_volume[2, 2, 5] = 3.0
    left = torch.arange(3 * 5 * 8, dtype=torch.float32).reshape(3, 5, 8)
    result = 提取稀疏双频证据(
        raw_volume,
        smooth_volume,
        left,
        np.array([5 * 1280 / 8], dtype=np.float32),
        np.array([2 * 288 / 5], dtype=np.float32),
        np.array([1 * 1280 / 8], dtype=np.float32),
    )
    assert result is not None
    assert result["原始误差"].tolist() == [0.0]
    assert result["平滑误差"].tolist() == [1.0]

    edge = np.linspace(0.0, 1.0, 200)
    raw = np.where(edge >= 0.5, 0.0, 2.0)
    smooth = np.where(edge < 0.5, 0.0, 2.0)
    fit = 拟合语义阈值(edge, raw, smooth, np.ones(200, dtype=bool))
    assert 0.45 <= fit["阈值"] <= 0.55

    regions = {region: np.ones(200, dtype=bool) for region in 区域顺序}
    records = {"边缘强度": edge, "原始误差": raw, "平滑误差": smooth, "区域": regions}
    summary, shares = 计算方法汇总(records, fit["阈值"])
    assert summary["上下文阈值门控"]["目标框内"]["平均绝对误差_bin"] == 0.0
    assert shares["目标框内选择原始比例"] > 0.4
    print("G4自检通过：稀疏双频曲线、样本外阈值门控和指标汇总逻辑正常")


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="versions/V09_V08O加点式三维质量排序/config.yaml")
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
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--points_per_region", type=int, default=384)
    parser.add_argument("--box_edge_width", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/选择性频率融合诊断_G4/G4_100样本"),
    )
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def 主程序(args: argparse.Namespace) -> int:
    if args.self_test:
        自检()
        return 0

    import torch
    import torch.nn.functional as F
    import tqdm
    from torch.utils.data import DataLoader, Subset

    from lib.helpers.config_helper import load_config
    from lib.helpers.qecr_dataloader_helper import build_qecr_dataloader
    from lib.helpers.qecr_model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint
    from lib.models.monodetr.depth_predictor.depth_predictor_lightstereo import correlation_volume

    if not torch.cuda.is_available():
        raise RuntimeError("G4完整诊断需要CUDA运行冻结V09")
    if args.num_samples < 2 or args.points_per_region <= 0:
        raise ValueError("样本数至少为2且每区点数必须为正")
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
    shuffled_indices = rng.choice(len(dataset), size=sample_count, replace=False).tolist()
    calibration_count = sample_count // 2
    calibration_ids = {
        str(dataset.idx_list[index]) for index in shuffled_indices[:calibration_count]
    }
    holdout_ids = {
        str(dataset.idx_list[index]) for index in shuffled_indices[calibration_count:]
    }
    selected_indices = sorted(shuffled_indices)
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=int(cfg["dataset"]["batch_size"]),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
    )

    model, _ = build_model(cfg["model"])
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    logger = logging.getLogger("StereoDETR_G4")
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
    calibration_records: List[Mapping[str, object]] = []
    holdout_records: List[Mapping[str, object]] = []
    failures: List[Dict[str, str]] = []
    calibration_success = 0
    holdout_success = 0
    progress = tqdm.tqdm(total=sample_count, desc="G4")

    try:
        with zipfile.ZipFile(args.velodyne_zip, "r") as archive:
            member_map = 构建点云成员映射(archive)
            with torch.no_grad():
                for inputs, calibs, targets, info in loader:
                    inputs = inputs.to(device)
                    calibs_gpu = calibs.to(device)
                    captured.clear()
                    _ = model(
                        inputs,
                        calibs_gpu,
                        targets,
                        info["img_size_croped"].to(device),
                        info["img_size_original"].to(device),
                        info["upper"].to(device),
                        dn_args=0,
                    )
                    features = captured.get("features")
                    if features is None:
                        raise RuntimeError("前向钩子没有捕获depth_predictor输入")
                    batch_size = inputs.shape[0]
                    s4 = features[0]
                    for batch_index in range(batch_size):
                        sample_id = str(info["img_id"][batch_index])
                        try:
                            calibration = dataset.get_calib(sample_id)
                            original_size = info["img_size_original"][batch_index].detach().cpu().numpy()
                            cloud = 从压缩包读取点云(archive, sample_id, member_map)
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
                                projection["x"],
                                projection["y"],
                                projection["z"],
                                targets,
                                batch_index,
                                input_width,
                                input_height,
                                args.box_edge_width,
                            )
                            projection, regions = 分层抽样(
                                projection, regions, args.points_per_region, rng
                            )
                            left = s4[batch_index]
                            right = s4[batch_index + batch_size]
                            raw_volume = correlation_volume(
                                left.unsqueeze(0), right.unsqueeze(0), 24
                            )[0]
                            smooth_volume = F.avg_pool2d(
                                raw_volume.unsqueeze(0), kernel_size=3, stride=1, padding=1
                            )[0]
                            evidence = 提取稀疏双频证据(
                                raw_volume,
                                smooth_volume,
                                left,
                                projection["x"],
                                projection["y"],
                                projection["disp"],
                            )
                            if evidence is None:
                                raise ValueError("没有满足s4双频约束的稀疏点")
                            valid_indices = evidence.pop("原始点索引")
                            record = {
                                **evidence,
                                "区域": {
                                    region: regions[region][valid_indices]
                                    for region in 区域顺序
                                },
                            }
                            if sample_id in calibration_ids:
                                calibration_records.append(record)
                                calibration_success += 1
                            elif sample_id in holdout_ids:
                                holdout_records.append(record)
                                holdout_success += 1
                            else:
                                raise RuntimeError("样本不属于校准或留出集合")
                        except Exception as error:
                            failures.append({"样本": sample_id, "错误": repr(error)})
                            print("[警告] {} 失败：{}".format(sample_id, error), file=sys.stderr)
                        progress.update(1)
    finally:
        hook.remove()
        progress.close()

    calibration_data = 合并记录(calibration_records)
    holdout_data = 合并记录(holdout_records)
    calibration = 拟合语义阈值(
        calibration_data["边缘强度"],
        calibration_data["原始误差"],
        calibration_data["平滑误差"],
        calibration_data["区域"]["目标框内"],
    )
    summary, shares = 计算方法汇总(holdout_data, calibration["阈值"])
    success_samples = calibration_success + holdout_success
    decision = 生成判断(summary, shares, success_samples)
    metadata = {
        "配置": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint),
        "velodyne_zip": str(args.velodyne_zip),
        "计划样本数": sample_count,
        "成功样本数": success_samples,
        "校准成功样本数": calibration_success,
        "留出成功样本数": holdout_success,
        "失败样本": failures,
        "随机种子": args.seed,
        "每区最多点数": args.points_per_region,
        "校准稀疏点数": int(len(calibration_data["边缘强度"])),
        "留出稀疏点数": int(len(holdout_data["边缘强度"])),
        "总耗时秒": time.time() - start,
    }
    写出结果(args.output_dir, summary, shares, calibration, decision, metadata)
    print("\nG4诊断完成")
    print("报告：{}".format(args.output_dir / "诊断报告.md"))
    print("自动结论：{}".format(decision["状态"]))
    print("解释：{}".format(decision["解释"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(主程序(解析参数()))
    except KeyboardInterrupt:
        print("用户中断", file=sys.stderr)
        raise SystemExit(130)
