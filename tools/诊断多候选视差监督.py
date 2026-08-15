#!/usr/bin/env python3
"""低成本诊断 StereoDETR 当前视差伪标签，并评估多候选监督的必要性。

本脚本不训练模型，也不修改数据集。它把 KITTI Velodyne 点投影到双目图像，
只把稀疏激光点作为离线评测参照，比较：

1. 当前 StereoDETR 的 StereoBM + 4x4 最大池化单值标签；
2. 同一 4x4 网格保留 16 个局部候选；
3. 修复配置失效后真正使用 StereoSGBM；
4. 左右一致性（LRC）过滤前后的候选质量。

注意：候选最小误差是“候选集合的覆盖能力”，不是模型实际预测误差。该指标只能
用于筛掉缺乏监督依据的方向，不能替代短程训练和完整消融。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np


方法顺序 = (
    "BM_单值最大池化",
    "BM_16候选",
    "BM_LRC_16候选",
    "SGBM_单值最大池化",
    "SGBM_16候选",
    "SGBM_LRC_16候选",
)

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


@dataclass
class 统计器:
    """在线累计计数和误差；误差分片仅用于最终精确中位数。"""

    总点数: int = 0
    有效点数: int = 0
    命中点数: int = 0
    候选总数: int = 0
    误差和: float = 0.0
    误差分片: List[np.ndarray] = field(default_factory=list)

    def 更新(
        self,
        组掩码: np.ndarray,
        有效掩码: np.ndarray,
        命中掩码: np.ndarray,
        误差: np.ndarray,
        候选数: np.ndarray,
    ) -> None:
        组掩码 = np.asarray(组掩码, dtype=bool)
        有效掩码 = np.asarray(有效掩码, dtype=bool) & 组掩码
        命中掩码 = np.asarray(命中掩码, dtype=bool) & 组掩码
        self.总点数 += int(组掩码.sum())
        self.有效点数 += int(有效掩码.sum())
        self.命中点数 += int(命中掩码.sum())
        self.候选总数 += int(np.asarray(候选数)[组掩码].sum())
        当前误差 = np.asarray(误差, dtype=np.float32)[有效掩码]
        if 当前误差.size:
            当前误差 = 当前误差[np.isfinite(当前误差)]
            if 当前误差.size:
                self.误差和 += float(当前误差.astype(np.float64).sum())
                self.误差分片.append(当前误差.copy())

    def 汇总(self) -> Dict[str, object]:
        if self.误差分片:
            所有误差 = np.concatenate(self.误差分片)
            中位数 = float(np.median(所有误差))
            p90 = float(np.percentile(所有误差, 90))
        else:
            中位数 = math.nan
            p90 = math.nan
        return {
            "总点数": self.总点数,
            "有效点数": self.有效点数,
            "有效率": 安全除法(self.有效点数, self.总点数),
            "支持命中点数": self.命中点数,
            "支持覆盖率": 安全除法(self.命中点数, self.总点数),
            "条件支持率": 安全除法(self.命中点数, self.有效点数),
            "平均绝对误差或最小候选误差_px": 安全除法(self.误差和, self.有效点数),
            "中位绝对误差或最小候选误差_px": 中位数,
            "P90绝对误差或最小候选误差_px": p90,
            "每点平均有效候选数": 安全除法(self.候选总数, self.总点数),
        }


def 安全除法(分子: float, 分母: float) -> float:
    return float(分子 / 分母) if 分母 else math.nan


def 读取标定(路径: Path) -> Dict[str, np.ndarray]:
    结果: Dict[str, np.ndarray] = {}
    for 行 in 路径.read_text(encoding="utf-8").splitlines():
        if ":" not in 行:
            continue
        键, 值 = 行.split(":", 1)
        数组 = np.fromstring(值, sep=" ", dtype=np.float64)
        if 键 in {"P2", "P3"}:
            结果[键] = 数组.reshape(3, 4)
        elif 键 in {"R0_rect", "R_rect"}:
            结果["R0_rect"] = 数组.reshape(3, 3)
        elif 键 in {"Tr_velo_to_cam", "Tr_velo_cam"}:
            结果["Tr_velo_to_cam"] = 数组.reshape(3, 4)
    缺少 = {"P2", "P3", "R0_rect", "Tr_velo_to_cam"} - set(结果)
    if 缺少:
        raise ValueError(f"标定文件 {路径} 缺少字段：{sorted(缺少)}")
    return 结果


def 从压缩包读取点云(
    压缩包: zipfile.ZipFile,
    样本号: str,
    成员映射: Mapping[str, str],
) -> np.ndarray:
    成员名 = 成员映射.get(样本号)
    if 成员名 is None:
        raise KeyError(f"Velodyne 压缩包中未找到 {样本号}.bin")
    数据 = 压缩包.read(成员名)
    点 = np.frombuffer(数据, dtype=np.float32)
    if 点.size % 4:
        raise ValueError(f"点云 {成员名} 的 float32 数量不是 4 的倍数")
    return 点.reshape(-1, 4)


def 构建点云成员映射(压缩包: zipfile.ZipFile) -> Dict[str, str]:
    映射: Dict[str, str] = {}
    for 名称 in 压缩包.namelist():
        if 名称.lower().endswith(".bin"):
            映射[Path(名称).stem] = 名称
    return 映射


def 投影点云到双目(
    点云: np.ndarray,
    标定: Mapping[str, np.ndarray],
    原图宽: int,
    原图高: int,
    目标宽: int,
    目标高: int,
    最大视差: float,
    顶部裁剪: int,
) -> Dict[str, np.ndarray]:
    """把 Velodyne 点投影到 P2/P3，并在目标分辨率下做像素级 z-buffer。"""

    点齐次 = np.concatenate(
        [点云[:, :3].astype(np.float64), np.ones((len(点云), 1))], axis=1
    )
    相机点 = 点齐次 @ 标定["Tr_velo_to_cam"].T
    矫正点 = 相机点 @ 标定["R0_rect"].T
    前方 = 矫正点[:, 2] > 1e-6
    矫正点 = 矫正点[前方]
    if not len(矫正点):
        return {键: np.empty(0, dtype=np.float32) for 键 in ("x", "y", "z", "disp")}

    矫正齐次 = np.concatenate(
        [矫正点, np.ones((len(矫正点), 1), dtype=np.float64)], axis=1
    )
    左投影 = 矫正齐次 @ 标定["P2"].T
    右投影 = 矫正齐次 @ 标定["P3"].T
    左x = 左投影[:, 0] / 左投影[:, 2]
    左y = 左投影[:, 1] / 左投影[:, 2]
    右x = 右投影[:, 0] / 右投影[:, 2]
    视差 = 左x - 右x

    横向比例 = 目标宽 / float(原图宽)
    纵向比例 = 目标高 / float(原图高)
    左x = 左x * 横向比例
    左y = 左y * 纵向比例
    视差 = 视差 * 横向比例
    深度 = 矫正点[:, 2]

    有效 = (
        np.isfinite(左x)
        & np.isfinite(左y)
        & np.isfinite(视差)
        & (左x >= 0)
        & (左x < 目标宽)
        & (左y >= 顶部裁剪)
        & (左y < 目标高)
        & (视差 > 0)
        & (视差 < 最大视差)
    )
    左x, 左y, 深度, 视差 = 左x[有效], 左y[有效], 深度[有效], 视差[有效]
    if not len(左x):
        return {键: np.empty(0, dtype=np.float32) for 键 in ("x", "y", "z", "disp")}

    # 相同整数像素只保留最近的激光点，避免后方表面重复计数。
    像素x = np.floor(左x).astype(np.int64)
    像素y = np.floor(左y).astype(np.int64)
    像素键 = 像素y * 目标宽 + 像素x
    排序 = np.lexsort((深度, 像素键))
    排序键 = 像素键[排序]
    保留 = np.ones(len(排序), dtype=bool)
    保留[1:] = 排序键[1:] != 排序键[:-1]
    索引 = 排序[保留]
    return {
        "x": 左x[索引].astype(np.float32),
        "y": 左y[索引].astype(np.float32),
        "z": 深度[索引].astype(np.float32),
        "disp": 视差[索引].astype(np.float32),
    }


def 创建匹配器(算法: str, 最大视差: int, 右匹配: bool = False):
    最小视差 = -最大视差 if 右匹配 else 0
    if 算法 == "BM":
        匹配器 = cv2.StereoBM_create(numDisparities=最大视差, blockSize=25)
        匹配器.setMinDisparity(最小视差)
        return 匹配器
    if 算法 == "SGBM":
        块大小 = 5
        return cv2.StereoSGBM_create(
            minDisparity=最小视差,
            numDisparities=最大视差,
            blockSize=块大小,
            P1=8 * 3 * 块大小**2,
            P2=32 * 3 * 块大小**2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
    raise ValueError(f"未知算法：{算法}")


def 计算左右视差(左灰度: np.ndarray, 右灰度: np.ndarray, 算法: str, 最大视差: int):
    左匹配器 = 创建匹配器(算法, 最大视差, 右匹配=False)
    右匹配器 = 创建匹配器(算法, 最大视差, 右匹配=True)
    左视差 = 左匹配器.compute(左灰度, 右灰度).astype(np.float32) / 16.0
    右视差 = 右匹配器.compute(右灰度, 左灰度).astype(np.float32) / 16.0
    左有效 = np.isfinite(左视差) & (左视差 > 0.0) & (左视差 < 最大视差)
    右有效 = np.isfinite(右视差) & (右视差 < 0.0) & (右视差 > -最大视差)
    左视差 = np.where(左有效, 左视差, np.nan).astype(np.float32)
    右视差 = np.where(右有效, 右视差, np.nan).astype(np.float32)
    return 左视差, 右视差


def 左右一致性掩码(左视差: np.ndarray, 右视差: np.ndarray, 阈值: float) -> np.ndarray:
    """检查 |d_left(x) + d_right(x-d_left)| <= threshold。"""

    高, 宽 = 左视差.shape
    x网格 = np.broadcast_to(np.arange(宽, dtype=np.float32), (高, 宽))
    y网格 = np.broadcast_to(np.arange(高, dtype=np.int64)[:, None], (高, 宽))
    右x = np.rint(x网格 - np.nan_to_num(左视差, nan=0.0)).astype(np.int64)
    边界内 = np.isfinite(左视差) & (右x >= 0) & (右x < 宽)
    安全右x = np.clip(右x, 0, 宽 - 1)
    对应右视差 = 右视差[y网格, 安全右x]
    return 边界内 & np.isfinite(对应右视差) & (np.abs(左视差 + 对应右视差) <= 阈值)


def 视差转网格候选(视差: np.ndarray, 网格大小: int) -> np.ndarray:
    """返回 [ceil(H/s), ceil(W/s), s*s]，越界填 NaN。"""

    高, 宽 = 视差.shape
    网格高 = math.ceil(高 / 网格大小)
    网格宽 = math.ceil(宽 / 网格大小)
    填充 = np.full((网格高 * 网格大小, 网格宽 * 网格大小), np.nan, np.float32)
    填充[:高, :宽] = 视差
    return (
        填充.reshape(网格高, 网格大小, 网格宽, 网格大小)
        .transpose(0, 2, 1, 3)
        .reshape(网格高, 网格宽, 网格大小 * 网格大小)
    )


def 候选生成单值(候选: np.ndarray) -> np.ndarray:
    有效 = np.isfinite(候选)
    单值 = np.full(候选.shape[:2], np.nan, dtype=np.float32)
    有值 = 有效.any(axis=-1)
    if 有值.any():
        替代 = np.where(有效, 候选, -np.inf)
        单值[有值] = 替代.max(axis=-1)[有值]
    return 单值


def 读取目标框(路径: Path, 横向比例: float, 纵向比例: float) -> List[Dict[str, object]]:
    目标框: List[Dict[str, object]] = []
    if not 路径.is_file():
        return 目标框
    for 行 in 路径.read_text(encoding="utf-8").splitlines():
        字段 = 行.split()
        if len(字段) < 8 or 字段[0] in {"DontCare", "Misc", "Tram"}:
            continue
        x1, y1, x2, y2 = map(float, 字段[4:8])
        目标框.append(
            {
                "类别": 字段[0],
                "框": np.array(
                    [x1 * 横向比例, y1 * 纵向比例, x2 * 横向比例, y2 * 纵向比例],
                    dtype=np.float32,
                ),
            }
        )
    return 目标框


def 点分组(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    目标框: Sequence[Mapping[str, object]],
    边界宽度: float,
) -> Dict[str, np.ndarray]:
    数量 = len(x)
    所属类别 = np.full(数量, "背景", dtype=object)
    最小面积 = np.full(数量, np.inf, dtype=np.float32)
    到边界距离 = np.full(数量, np.inf, dtype=np.float32)
    for 项 in 目标框:
        x1, y1, x2, y2 = np.asarray(项["框"], dtype=np.float32)
        内部 = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
        面积 = max(float((x2 - x1) * (y2 - y1)), 0.0)
        采用 = 内部 & (面积 < 最小面积)
        if not 采用.any():
            continue
        最小面积[采用] = 面积
        所属类别[采用] = str(项["类别"])
        到边界距离[采用] = np.minimum.reduce(
            [x[采用] - x1, x2 - x[采用], y[采用] - y1, y2 - y[采用]]
        )
    框内 = 所属类别 != "背景"
    边界 = 框内 & (到边界距离 <= 边界宽度)
    return {
        "全部": np.ones(数量, dtype=bool),
        "目标框内": 框内,
        "目标边界": 边界,
        "目标内部": 框内 & ~边界,
        "背景": ~框内,
        "近距_<20m": z < 20.0,
        "中距_20-40m": (z >= 20.0) & (z < 40.0),
        "远距_>=40m": z >= 40.0,
        "类别_Car": 所属类别 == "Car",
        "类别_Pedestrian": 所属类别 == "Pedestrian",
        "类别_Cyclist": 所属类别 == "Cyclist",
    }


def 从网格取候选(候选网格: np.ndarray, x: np.ndarray, y: np.ndarray, 网格大小: int) -> np.ndarray:
    网格x = np.floor(x / 网格大小).astype(np.int64)
    网格y = np.floor(y / 网格大小).astype(np.int64)
    网格x = np.clip(网格x, 0, 候选网格.shape[1] - 1)
    网格y = np.clip(网格y, 0, 候选网格.shape[0] - 1)
    return 候选网格[网格y, 网格x]


def 计算方法点指标(
    候选: np.ndarray,
    真值: np.ndarray,
    支持半径: float,
    单值模式: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if 单值模式:
        if 候选.ndim == 2:
            候选 = 候选[:, :1]
        有效 = np.isfinite(候选[:, 0])
        误差 = np.full(len(真值), np.nan, dtype=np.float32)
        误差[有效] = np.abs(候选[有效, 0] - 真值[有效])
        候选数 = 有效.astype(np.int32)
    else:
        候选有效 = np.isfinite(候选)
        候选数 = 候选有效.sum(axis=1).astype(np.int32)
        有效 = 候选数 > 0
        差 = np.where(候选有效, np.abs(候选 - 真值[:, None]), np.inf)
        误差 = 差.min(axis=1).astype(np.float32)
        误差[~有效] = np.nan
    命中 = 有效 & (误差 <= 支持半径)
    return 有效, 命中, 误差, 候选数


def 选择样本(全部样本: Sequence[str], 数量: int, 随机种子: int) -> List[str]:
    if 数量 <= 0 or 数量 >= len(全部样本):
        return list(全部样本)
    随机 = random.Random(随机种子)
    return sorted(随机.sample(list(全部样本), 数量))


def 转为可序列化(对象):
    if isinstance(对象, dict):
        return {str(k): 转为可序列化(v) for k, v in 对象.items()}
    if isinstance(对象, (list, tuple)):
        return [转为可序列化(v) for v in 对象]
    if isinstance(对象, np.generic):
        return 对象.item()
    if isinstance(对象, float) and not math.isfinite(对象):
        return None
    return 对象


def 百分数(值: Optional[float]) -> str:
    if 值 is None or not math.isfinite(float(值)):
        return "—"
    return f"{100.0 * float(值):.2f}%"


def 小数(值: Optional[float], 位数: int = 3) -> str:
    if 值 is None or not math.isfinite(float(值)):
        return "—"
    return f"{float(值):.{位数}f}"


def 生成判断(
    汇总: Mapping[str, Mapping[str, Mapping[str, object]]],
    实际样本数: int,
) -> Dict[str, object]:
    def 取(方法: str, 分组: str, 指标: str) -> float:
        值 = 汇总.get(方法, {}).get(分组, {}).get(指标, math.nan)
        return float(值) if 值 is not None else math.nan

    核心组 = "目标边界"
    if 取("BM_16候选", 核心组, "总点数") < 100:
        核心组 = "目标框内"
    bm单覆盖 = 取("BM_单值最大池化", 核心组, "支持覆盖率")
    bm多覆盖 = 取("BM_16候选", 核心组, "支持覆盖率")
    bm单误差 = 取("BM_单值最大池化", 核心组, "平均绝对误差或最小候选误差_px")
    bm多误差 = 取("BM_16候选", 核心组, "平均绝对误差或最小候选误差_px")
    bm覆盖提升 = bm多覆盖 - bm单覆盖
    bm误差改善 = bm单误差 - bm多误差

    sgbm单覆盖 = 取("SGBM_单值最大池化", 核心组, "支持覆盖率")
    sgbm多覆盖 = 取("SGBM_16候选", 核心组, "支持覆盖率")
    sgbm单误差 = 取("SGBM_单值最大池化", 核心组, "平均绝对误差或最小候选误差_px")
    sgbm多误差 = 取("SGBM_16候选", 核心组, "平均绝对误差或最小候选误差_px")
    sgbm多候选覆盖提升 = sgbm多覆盖 - sgbm单覆盖
    sgbm多候选误差改善 = sgbm单误差 - sgbm多误差

    sgbm覆盖差 = 取("SGBM_16候选", 核心组, "支持覆盖率") - bm多覆盖
    sgbm误差差 = bm多误差 - 取("SGBM_16候选", 核心组, "平均绝对误差或最小候选误差_px")
    sgbm有效差 = 取("SGBM_16候选", 核心组, "有效率") - 取("BM_16候选", 核心组, "有效率")
    if np.isfinite(sgbm覆盖差) and np.isfinite(sgbm误差差) and sgbm有效差 >= -0.03 and (sgbm覆盖差 >= 0.01 or sgbm误差差 >= 0.10):
        sgbm判断 = "优先 SGBM：在没有明显损失有效率时改善了候选质量。"
        sgbm状态 = "优先SGBM"
    else:
        sgbm判断 = "保留 BM：当前 SGBM 参数没有形成稳定净收益；只修复配置失效，不把 SGBM 当创新点。"
        sgbm状态 = "保留BM"

    # 多候选必须在最终建议采用的匹配器内部比较，不能只看 BM。
    候选判断匹配器 = "SGBM" if sgbm状态 == "优先SGBM" else "BM"
    if 候选判断匹配器 == "SGBM":
        覆盖提升 = sgbm多候选覆盖提升
        误差改善 = sgbm多候选误差改善
    else:
        覆盖提升 = bm覆盖提升
        误差改善 = bm误差改善
    if np.isfinite(覆盖提升) and np.isfinite(误差改善) and 覆盖提升 >= 0.05 and 误差改善 >= 0.5:
        mc判断 = (
            f"通过：在 {候选判断匹配器} 内，多候选明显修复了单值最大池化的"
            "信息丢失，可进入短程训练消融。"
        )
        mc状态 = "通过"
    elif np.isfinite(覆盖提升) and np.isfinite(误差改善) and 覆盖提升 >= 0.02 and 误差改善 >= 0.2:
        mc判断 = (
            f"谨慎：在 {候选判断匹配器} 内存在可测收益，但幅度不足以直接投入"
            "完整训练，先做10–15轮配对实验。"
        )
        mc状态 = "谨慎"
    else:
        mc判断 = (
            f"否决：在建议采用的 {候选判断匹配器} 内仍缺乏足够离线证据，"
            "暂不实现多候选正式版本。"
        )
        mc状态 = "否决"

    lrc基础方法 = f"{候选判断匹配器}_16候选"
    lrc方法 = f"{候选判断匹配器}_LRC_16候选"
    lrc误差改善 = 取(lrc基础方法, 核心组, "平均绝对误差或最小候选误差_px") - 取(
        lrc方法, 核心组, "平均绝对误差或最小候选误差_px"
    )
    lrc有效损失 = 取(lrc基础方法, 核心组, "有效率") - 取(lrc方法, 核心组, "有效率")
    lrc覆盖损失 = 取(lrc基础方法, 核心组, "支持覆盖率") - 取(
        lrc方法, 核心组, "支持覆盖率"
    )
    if (
        np.isfinite(lrc误差改善)
        and np.isfinite(lrc有效损失)
        and np.isfinite(lrc覆盖损失)
        and lrc误差改善 >= 0.20
        and lrc有效损失 <= 0.10
        and lrc覆盖损失 <= 0.005
    ):
        lrc判断 = "可用硬 LRC：误差下降，且有效率与全点支持覆盖损失均可控。"
        lrc状态 = "硬过滤可用"
    elif np.isfinite(lrc误差改善) and lrc误差改善 > 0:
        lrc判断 = (
            "只作软置信度：LRC 能降低条件误差，但硬过滤会降低有效率或全点支持覆盖。"
        )
        lrc状态 = "仅软权重"
    else:
        lrc判断 = "暂不使用 LRC：未观察到可靠去噪收益。"
        lrc状态 = "不使用"

    if 实际样本数 < 100:
        判断等级 = "冒烟预观察"
        mc状态 = f"冒烟预观察：{mc状态}"
        sgbm状态 = f"冒烟预观察：{sgbm状态}"
        lrc状态 = f"冒烟预观察：{lrc状态}"
        样本提示 = (
            f"当前只有 {实际样本数} 个样本，只验证流程并提供预观察，"
            "不能据此批准或否决正式版本；请以不少于100个、建议200个样本为准。"
        )
        mc判断 = 样本提示 + mc判断
        sgbm判断 = 样本提示 + sgbm判断
        lrc判断 = 样本提示 + lrc判断
    else:
        判断等级 = "正式G0"

    return {
        "判断等级": 判断等级,
        "核心判断分组": 核心组,
        "候选判断匹配器": 候选判断匹配器,
        "多候选状态": mc状态,
        "多候选结论": mc判断,
        "候选匹配器内多候选相对单值覆盖提升_百分点": 覆盖提升 * 100.0,
        "候选匹配器内多候选相对单值误差改善_px": 误差改善,
        "BM多候选相对单值覆盖提升_百分点": bm覆盖提升 * 100.0,
        "BM多候选相对单值误差改善_px": bm误差改善,
        "SGBM多候选相对单值覆盖提升_百分点": sgbm多候选覆盖提升 * 100.0,
        "SGBM多候选相对单值误差改善_px": sgbm多候选误差改善,
        "匹配器状态": sgbm状态,
        "匹配器结论": sgbm判断,
        "SGBM相对BM覆盖变化_百分点": sgbm覆盖差 * 100.0,
        "SGBM相对BM误差改善_px": sgbm误差差,
        "LRC状态": lrc状态,
        "LRC结论": lrc判断,
        "LRC误差改善_px": lrc误差改善,
        "LRC有效率损失_百分点": lrc有效损失 * 100.0,
        "LRC支持覆盖率损失_百分点": lrc覆盖损失 * 100.0,
    }


def 写出报告(
    输出目录: Path,
    汇总: Mapping[str, Mapping[str, Mapping[str, object]]],
    判断: Mapping[str, object],
    元数据: Mapping[str, object],
) -> None:
    输出目录.mkdir(parents=True, exist_ok=True)
    csv路径 = 输出目录 / "诊断汇总.csv"
    字段 = [
        "方法",
        "分组",
        "总点数",
        "有效点数",
        "有效率",
        "支持命中点数",
        "支持覆盖率",
        "条件支持率",
        "平均绝对误差或最小候选误差_px",
        "中位绝对误差或最小候选误差_px",
        "P90绝对误差或最小候选误差_px",
        "每点平均有效候选数",
    ]
    with csv路径.open("w", newline="", encoding="utf-8-sig") as 文件:
        写入器 = csv.DictWriter(文件, fieldnames=字段)
        写入器.writeheader()
        for 方法 in 方法顺序:
            for 分组 in 分组顺序:
                行 = {"方法": 方法, "分组": 分组}
                行.update(汇总[方法][分组])
                写入器.writerow(转为可序列化(行))

    完整结果 = {"元数据": 元数据, "自动判断": 判断, "汇总": 汇总}
    (输出目录 / "诊断明细.json").write_text(
        json.dumps(转为可序列化(完整结果), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    核心组 = str(判断["核心判断分组"])
    表格行 = []
    for 方法 in 方法顺序:
        结果 = 汇总[方法][核心组]
        表格行.append(
            "| {方法} | {有效率} | {覆盖率} | {条件率} | {误差} | {中位} | {候选数} |".format(
                方法=方法,
                有效率=百分数(结果["有效率"]),
                覆盖率=百分数(结果["支持覆盖率"]),
                条件率=百分数(结果["条件支持率"]),
                误差=小数(结果["平均绝对误差或最小候选误差_px"]),
                中位=小数(结果["中位绝对误差或最小候选误差_px"]),
                候选数=小数(结果["每点平均有效候选数"], 2),
            )
        )

    距离行 = []
    for 分组 in ("近距_<20m", "中距_20-40m", "远距_>=40m", "目标边界"):
        单值 = 汇总["BM_单值最大池化"][分组]
        多值 = 汇总["BM_16候选"][分组]
        距离行.append(
            "| {分组} | {点数} | {单覆盖} | {多覆盖} | {变化} | {单误差} | {多误差} |".format(
                分组=分组,
                点数=单值["总点数"],
                单覆盖=百分数(单值["支持覆盖率"]),
                多覆盖=百分数(多值["支持覆盖率"]),
                变化=小数(100.0 * (float(多值["支持覆盖率"]) - float(单值["支持覆盖率"])), 2),
                单误差=小数(单值["平均绝对误差或最小候选误差_px"]),
                多误差=小数(多值["平均绝对误差或最小候选误差_px"]),
            )
        )

    markdown = f"""# StereoDETR 多候选视差监督 G0 诊断报告

## 结论先行

- 判断等级：**{判断['判断等级']}**。少于100个样本时只能作为管线冒烟和趋势预观察。
- 多候选：**{判断['多候选状态']}**。{判断['多候选结论']}
- 匹配器：**{判断['匹配器状态']}**。{判断['匹配器结论']}
- 左右一致性：**{判断['LRC状态']}**。{判断['LRC结论']}
- 本报告只证明监督标签的离线质量，不证明 3D AP 必然提升。若多候选通过，也必须先做同种子、同起点的 10–15 轮配对训练。

## 实验设置

- 样本数：{元数据['实际完成样本数']} / {元数据['计划样本数']}
- 稀疏参照点数：{元数据['有效投影点总数']}
- 输入分辨率：{元数据['目标宽']} × {元数据['目标高']}，顶部排除 {元数据['顶部裁剪']} px
- 网格：{元数据['网格大小']} × {元数据['网格大小']}，每格最多 {int(元数据['网格大小']) ** 2} 个候选
- 支持半径：±{元数据['支持半径_px']} px；LRC 阈值：{元数据['LRC阈值_px']} px
- 点云只从 ZIP 中按样本读取，没有完整解压，也没有参与模型训练。

## 核心分组：{核心组}

| 方法 | 有效率 | 支持覆盖率 | 条件支持率 | 平均误差/最小候选误差(px) | 中位误差(px) | 平均有效候选数 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(表格行)}

“单值”的误差是实际单标签误差；“16候选”的误差是集合内最接近参照值的 oracle 最小误差。两者用途不同，不能把候选最小误差当成网络预测 EPE。

## BM 单值与多候选的关键分层

| 分组 | 点数 | 单值覆盖率 | 多候选覆盖率 | 覆盖变化(百分点) | 单值误差(px) | 候选最小误差(px) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(距离行)}

## 自动门槛的具体数值

- 候选判断采用：{判断['候选判断匹配器']}。
- 该匹配器内多候选覆盖提升：{小数(判断['候选匹配器内多候选相对单值覆盖提升_百分点'], 2)} 个百分点；误差改善 {小数(判断['候选匹配器内多候选相对单值误差改善_px'])} px。
- BM 多候选覆盖提升：{小数(判断['BM多候选相对单值覆盖提升_百分点'], 2)} 个百分点。
- BM 多候选误差改善：{小数(判断['BM多候选相对单值误差改善_px'])} px。
- SGBM 多候选覆盖提升：{小数(判断['SGBM多候选相对单值覆盖提升_百分点'], 2)} 个百分点。
- SGBM 多候选误差改善：{小数(判断['SGBM多候选相对单值误差改善_px'])} px。
- SGBM 相对 BM 覆盖变化：{小数(判断['SGBM相对BM覆盖变化_百分点'], 2)} 个百分点。
- SGBM 相对 BM 误差改善：{小数(判断['SGBM相对BM误差改善_px'])} px。
- LRC 误差改善：{小数(判断['LRC误差改善_px'])} px；有效率损失 {小数(判断['LRC有效率损失_百分点'], 2)} 个百分点；全点支持覆盖损失 {小数(判断['LRC支持覆盖率损失_百分点'], 2)} 个百分点。

## 如何据此决定下一步

1. 多候选为“通过”：实现 V05-MC-PMC，但先从 V00 最优权重做 10–15 轮配对续训；不要立即跑 195 轮。
2. 多候选为“谨慎”：先缩小实现，只改标签与损失，不加新推理分支；短程结果至少提升 Car 3D AP_R40 Moderate 0.3 才进入完整训练。
3. 多候选为“否决”：停止该路线，回到误差归因，不继续租卡验证它。
4. LRC 为“仅软权重”：后续只能把一致性残差转成训练权重，不能硬删除全部不一致候选。

## 局限

- Velodyne 是稀疏评测参照，尤其行人、骑行者和物体边缘点较少。
- “目标边界”按 KITTI 2D 框边缘近似，不等于真实实例轮廓。
- 这里只测无随机裁剪/翻转的规范图像；训练增强后的标签质量需要在 V05 单元测试中继续验证。
- 候选覆盖率提高只表示正确监督更可能被保留，最终 AP 仍取决于候选概率质量、损失归一化和优化稳定性。

原始机器可读结果见 `诊断汇总.csv` 与 `诊断明细.json`。
"""
    (输出目录 / "诊断报告.md").write_text(markdown, encoding="utf-8")


def 自检() -> None:
    网格 = 视差转网格候选(np.arange(16, dtype=np.float32).reshape(4, 4), 4)
    assert 网格.shape == (1, 1, 16)
    assert np.array_equal(np.sort(网格[0, 0]), np.arange(16, dtype=np.float32))
    单值 = 候选生成单值(网格)
    assert float(单值[0, 0]) == 15.0

    左 = np.full((1, 6), np.nan, dtype=np.float32)
    右 = np.full((1, 6), np.nan, dtype=np.float32)
    左[0, 4] = 2.0
    右[0, 2] = -2.0
    assert bool(左右一致性掩码(左, 右, 0.1)[0, 4])
    右[0, 2] = -1.0
    assert not bool(左右一致性掩码(左, 右, 0.1)[0, 4])

    标定 = {
        "P2": np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 1, 0]], np.float64),
        "P3": np.array([[10, 0, 0, -5], [0, 10, 0, 0], [0, 0, 1, 0]], np.float64),
        "R0_rect": np.eye(3, dtype=np.float64),
        "Tr_velo_to_cam": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], np.float64),
    }
    点 = np.array([[1, 1, 10, 0]], dtype=np.float32)
    投影 = 投影点云到双目(点, 标定, 20, 20, 20, 20, 10, 0)
    assert len(投影["disp"]) == 1
    assert abs(float(投影["disp"][0]) - 0.5) < 1e-6

    候选 = np.array([[1.0, 3.0, np.nan], [np.nan, np.nan, np.nan]], np.float32)
    有效, 命中, 误差, 数量 = 计算方法点指标(候选, np.array([2.8, 1.0], np.float32), 0.5, False)
    assert 有效.tolist() == [True, False]
    assert 命中.tolist() == [True, False]
    assert abs(float(误差[0]) - 0.2) < 1e-5
    assert 数量.tolist() == [2, 0]

    # 实际调用左右 BM/SGBM，防止负 minDisparity 在目标 OpenCV 版本中不受支持。
    随机 = np.random.default_rng(7)
    左灰 = 随机.integers(0, 256, size=(64, 128), dtype=np.uint8)
    右灰 = np.roll(左灰, -4, axis=1)
    for 算法 in ("BM", "SGBM"):
        左结果, 右结果 = 计算左右视差(左灰, 右灰, 算法, 16)
        assert 左结果.shape == 左灰.shape
        assert 右结果.shape == 右灰.shape
        一致 = 左右一致性掩码(左结果, 右结果, 1.0)
        assert 一致.shape == 左灰.shape

    # 用首轮20样本的量级验证判据：SGBM 内多候选应为“谨慎”，LRC只能软加权。
    预观察 = {
        "BM_单值最大池化": (0.6479, 0.1431, 9.452),
        "BM_16候选": (0.6479, 0.1526, 9.306),
        "BM_LRC_16候选": (0.5617, 0.1389, 8.359),
        "SGBM_单值最大池化": (0.8516, 0.1751, 10.279),
        "SGBM_16候选": (0.8516, 0.2214, 9.018),
        "SGBM_LRC_16候选": (0.7869, 0.2028, 8.542),
    }
    模拟汇总 = {}
    for 方法, (有效率, 覆盖率, 平均误差) in 预观察.items():
        模拟汇总[方法] = {
            "目标边界": {
                "总点数": 13912,
                "有效率": 有效率,
                "支持覆盖率": 覆盖率,
                "平均绝对误差或最小候选误差_px": 平均误差,
            }
        }
    冒烟判断 = 生成判断(模拟汇总, 20)
    assert 冒烟判断["候选判断匹配器"] == "SGBM"
    assert 冒烟判断["多候选状态"] == "冒烟预观察：谨慎"
    assert 冒烟判断["LRC状态"] == "冒烟预观察：仅软权重"
    正式判断 = 生成判断(模拟汇总, 200)
    assert 正式判断["多候选状态"] == "谨慎"
    assert 正式判断["LRC状态"] == "仅软权重"
    print("G0 自检通过：网格候选、最大池化、LRC、投影和指标逻辑均正常")


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/KITTI/object/training"),
        help="包含 image_2/image_3/calib/label_2/ImageSets 的 KITTI training 目录",
    )
    parser.add_argument(
        "--velodyne_zip",
        type=Path,
        default=Path("/autodl-pub/data/KITTI_Object/raw/data_object_velodyne.zip"),
    )
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--target_width", type=int, default=1280)
    parser.add_argument("--target_height", type=int, default=388)
    parser.add_argument("--crop_top", type=int, default=100)
    parser.add_argument("--cell_size", type=int, default=4)
    parser.add_argument("--support_radius", type=float, default=2.0)
    parser.add_argument("--lrc_threshold", type=float, default=1.0)
    parser.add_argument("--box_edge_width", type=float, default=8.0)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("outputs/视差监督诊断_G0")
    )
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def 主程序(参数: argparse.Namespace) -> int:
    if 参数.self_test:
        自检()
        return 0
    if 参数.max_disp <= 0 or 参数.max_disp % 16:
        raise ValueError("--max_disp 必须是正数且为 16 的倍数")
    if 参数.cell_size <= 0:
        raise ValueError("--cell_size 必须为正数")
    if 参数.crop_top < 0 or 参数.crop_top >= 参数.target_height:
        raise ValueError("--crop_top 必须位于 [0, target_height) 内")

    数据根目录 = 参数.data_root.resolve()
    划分文件 = 参数.split_file or (数据根目录 / "ImageSets" / "train.txt")
    必需路径 = [
        数据根目录 / "image_2",
        数据根目录 / "image_3",
        数据根目录 / "calib",
        数据根目录 / "label_2",
        划分文件,
        参数.velodyne_zip,
    ]
    缺少 = [str(路径) for 路径 in 必需路径 if not 路径.exists()]
    if 缺少:
        raise FileNotFoundError("缺少以下输入：\n" + "\n".join(缺少))

    全部样本 = [行.strip() for 行 in 划分文件.read_text(encoding="utf-8").splitlines() if 行.strip()]
    样本 = 选择样本(全部样本, 参数.num_samples, 参数.seed)
    if not 样本:
        raise ValueError(f"划分文件 {划分文件} 没有样本")

    累计: MutableMapping[str, MutableMapping[str, 统计器]] = defaultdict(
        lambda: defaultdict(统计器)
    )
    失败: List[Dict[str, str]] = []
    完成数 = 0
    总投影点 = 0
    开始 = time.time()

    print(f"计划诊断 {len(样本)} 个样本；从 ZIP 按需读取点云，不做完整解压", flush=True)
    with zipfile.ZipFile(参数.velodyne_zip, "r") as 压缩包:
        成员映射 = 构建点云成员映射(压缩包)
        for 序号, 样本号 in enumerate(样本, 1):
            try:
                左图路径 = 数据根目录 / "image_2" / f"{样本号}.png"
                右图路径 = 数据根目录 / "image_3" / f"{样本号}.png"
                左图 = cv2.imread(str(左图路径), cv2.IMREAD_COLOR)
                右图 = cv2.imread(str(右图路径), cv2.IMREAD_COLOR)
                if 左图 is None or 右图 is None:
                    raise FileNotFoundError(f"图像读取失败：{左图路径} / {右图路径}")
                原高, 原宽 = 左图.shape[:2]
                if 右图.shape[:2] != (原高, 原宽):
                    raise ValueError("左右图尺寸不同")
                左图 = cv2.resize(左图, (参数.target_width, 参数.target_height), interpolation=cv2.INTER_LINEAR)
                右图 = cv2.resize(右图, (参数.target_width, 参数.target_height), interpolation=cv2.INTER_LINEAR)
                左灰 = cv2.cvtColor(左图, cv2.COLOR_BGR2GRAY)
                右灰 = cv2.cvtColor(右图, cv2.COLOR_BGR2GRAY)

                标定 = 读取标定(数据根目录 / "calib" / f"{样本号}.txt")
                点云 = 从压缩包读取点云(压缩包, 样本号, 成员映射)
                投影 = 投影点云到双目(
                    点云,
                    标定,
                    原宽,
                    原高,
                    参数.target_width,
                    参数.target_height,
                    参数.max_disp,
                    参数.crop_top,
                )
                if not len(投影["disp"]):
                    raise ValueError("没有有效投影点")
                横向比例 = 参数.target_width / float(原宽)
                纵向比例 = 参数.target_height / float(原高)
                目标框 = 读取目标框(
                    数据根目录 / "label_2" / f"{样本号}.txt", 横向比例, 纵向比例
                )
                分组 = 点分组(
                    投影["x"], 投影["y"], 投影["z"], 目标框, 参数.box_edge_width
                )

                方法候选: Dict[str, Tuple[np.ndarray, bool]] = {}
                for 算法 in ("BM", "SGBM"):
                    左视差, 右视差 = 计算左右视差(左灰, 右灰, 算法, 参数.max_disp)
                    一致 = 左右一致性掩码(左视差, 右视差, 参数.lrc_threshold)
                    左视差LRC = np.where(一致, 左视差, np.nan).astype(np.float32)
                    原候选网格 = 视差转网格候选(左视差, 参数.cell_size)
                    LRC候选网格 = 视差转网格候选(左视差LRC, 参数.cell_size)
                    单值网格 = 候选生成单值(原候选网格)
                    点候选 = 从网格取候选(
                        原候选网格, 投影["x"], 投影["y"], 参数.cell_size
                    )
                    点单值 = 从网格取候选(
                        单值网格[..., None], 投影["x"], 投影["y"], 参数.cell_size
                    )
                    点LRC候选 = 从网格取候选(
                        LRC候选网格, 投影["x"], 投影["y"], 参数.cell_size
                    )
                    方法候选[f"{算法}_单值最大池化"] = (点单值, True)
                    方法候选[f"{算法}_16候选"] = (点候选, False)
                    方法候选[f"{算法}_LRC_16候选"] = (点LRC候选, False)

                for 方法, (候选, 单值模式) in 方法候选.items():
                    有效, 命中, 误差, 候选数 = 计算方法点指标(
                        候选, 投影["disp"], 参数.support_radius, 单值模式
                    )
                    for 分组名 in 分组顺序:
                        累计[方法][分组名].更新(
                            分组[分组名], 有效, 命中, 误差, 候选数
                        )
                完成数 += 1
                总投影点 += len(投影["disp"])
            except Exception as 异常:  # 单样本失败不浪费整个诊断
                失败.append({"样本": 样本号, "错误": repr(异常)})
                print(f"[警告] {样本号} 失败：{异常}", file=sys.stderr, flush=True)

            if 序号 == 1 or 序号 % 10 == 0 or 序号 == len(样本):
                已用 = time.time() - 开始
                print(
                    f"进度 {序号}/{len(样本)}，成功 {完成数}，失败 {len(失败)}，"
                    f"累计参照点 {总投影点}，耗时 {已用/60:.1f} 分钟",
                    flush=True,
                )

    if 完成数 == 0:
        raise RuntimeError("所有样本均失败，未生成报告")
    汇总: Dict[str, Dict[str, Dict[str, object]]] = {}
    for 方法 in 方法顺序:
        汇总[方法] = {}
        for 分组名 in 分组顺序:
            汇总[方法][分组名] = 累计[方法][分组名].汇总()
    判断 = 生成判断(汇总, 完成数)
    元数据 = {
        "data_root": str(数据根目录),
        "velodyne_zip": str(参数.velodyne_zip),
        "split_file": str(划分文件),
        "随机种子": 参数.seed,
        "计划样本数": len(样本),
        "实际完成样本数": 完成数,
        "失败样本": 失败,
        "有效投影点总数": 总投影点,
        "最大视差": 参数.max_disp,
        "目标宽": 参数.target_width,
        "目标高": 参数.target_height,
        "顶部裁剪": 参数.crop_top,
        "网格大小": 参数.cell_size,
        "支持半径_px": 参数.support_radius,
        "LRC阈值_px": 参数.lrc_threshold,
        "目标框边缘宽度_px": 参数.box_edge_width,
        "总耗时秒": time.time() - 开始,
    }
    写出报告(参数.output_dir, 汇总, 判断, 元数据)
    print("\n诊断完成：", flush=True)
    print(f"  Markdown：{参数.output_dir / '诊断报告.md'}", flush=True)
    print(f"  CSV：{参数.output_dir / '诊断汇总.csv'}", flush=True)
    print(f"  JSON：{参数.output_dir / '诊断明细.json'}", flush=True)
    print(f"  多候选结论：{判断['多候选状态']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(主程序(解析参数()))
    except KeyboardInterrupt:
        print("\n用户中止诊断", file=sys.stderr)
        raise SystemExit(130)
