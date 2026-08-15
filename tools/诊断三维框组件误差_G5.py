#!/usr/bin/env python3
"""G5：使用 V09 已保存的 KITTI 预测做三维框组件误差归因。

本脚本不训练、不加载模型，也不修改 checkpoint。它先按照类别和 2D IoU 做
一对一匹配，再分别用验证集真值替换预测框的轴向深度、投影中心、完整位置、
尺寸和朝向，最后调用项目自带的 KITTI 官方评估代码计算 3D AP_R40 上限。

所有替换实验都使用验证集真值，只能作为诊断 Oracle，绝不能作为模型结果、
论文主表结果或 KITTI 提交结果。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


项目根目录 = Path(__file__).resolve().parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))


类别编号 = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

组件变体 = (
    "GT轴向深度_保留预测射线",
    "GT投影中心_保留预测深度",
    "GT完整三维位置",
    "GT三维尺寸",
    "GT朝向",
    "GT全部三维属性",
)


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G5：V09 三维框组件误差归因（只读 Oracle 诊断）"
    )
    parser.add_argument(
        "--config",
        default="versions/V09_V08O加点式三维质量排序/config.yaml",
        help="用于解析 V09 输出目录和 KITTI 验证集路径的配置",
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help="V09 已保存的 outputs/data；默认从配置自动推导",
    )
    parser.add_argument("--label_dir", default=None)
    parser.add_argument("--calib_dir", default=None)
    parser.add_argument("--split_file", default=None)
    parser.add_argument(
        "--output_dir",
        default="outputs/三维框组件误差归因_G5",
    )
    parser.add_argument(
        "--match_iou_2d",
        type=float,
        default=0.30,
        help="同类预测与 GT 做匈牙利匹配后的最低 2D IoU",
    )
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def 规范样本号(value: object) -> str:
    if hasattr(value, "item"):
        value = value.item()
    text = str(value)
    if text.isdigit():
        return f"{int(text):06d}"
    return text


def 读取样本号(path: Path) -> List[str]:
    values = [
        规范样本号(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"划分文件为空：{path}")
    if len(values) != len(set(values)):
        raise ValueError(f"划分文件含重复样本：{path}")
    return values


def 二维IoU矩阵(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    boxes1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    if not len(boxes1) or not len(boxes2):
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    left_top = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.maximum(right_bottom - left_top, 0.0)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0.0) * np.maximum(
        boxes1[:, 3] - boxes1[:, 1], 0.0
    )
    area2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0.0) * np.maximum(
        boxes2[:, 3] - boxes2[:, 1], 0.0
    )
    union = area1[:, None] + area2[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def 一对一二维匹配(
    gt: Mapping[str, np.ndarray],
    dt: Mapping[str, np.ndarray],
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """返回 ``(检测索引, GT索引, 2D IoU)``，匹配过程不使用任何 3D 属性。"""

    from scipy.optimize import linear_sum_assignment

    matches: List[Tuple[int, int, float]] = []
    for class_name in 类别编号:
        gt_index = np.flatnonzero(np.asarray(gt["name"]) == class_name)
        dt_index = np.flatnonzero(np.asarray(dt["name"]) == class_name)
        if not len(gt_index) or not len(dt_index):
            continue
        iou = 二维IoU矩阵(
            np.asarray(dt["bbox"])[dt_index],
            np.asarray(gt["bbox"])[gt_index],
        )
        row, col = linear_sum_assignment(1.0 - iou)
        for local_dt, local_gt in zip(row.tolist(), col.tolist()):
            value = float(iou[local_dt, local_gt])
            if value >= threshold:
                matches.append(
                    (int(dt_index[local_dt]), int(gt_index[local_gt]), value)
                )
    return matches


def 角度差(angle1: float, angle2: float) -> float:
    return abs((float(angle1) - float(angle2) + math.pi) % (2.0 * math.pi) - math.pi)


def 三维中心(location: np.ndarray, dimensions_lhw: np.ndarray) -> np.ndarray:
    center = np.asarray(location, dtype=np.float64).copy()
    # KITTI location 是底面中心；项目解析后的 dimensions 顺序为 [l, h, w]。
    center[1] -= float(dimensions_lhw[1]) / 2.0
    return center


def 投影中心(calib: object, location: np.ndarray, dimensions_lhw: np.ndarray) -> np.ndarray:
    center = 三维中心(location, dimensions_lhw)
    point, _ = calib.rect_to_img(center.reshape(1, 3))
    return np.asarray(point[0], dtype=np.float64)


def 从投影中心恢复底面位置(
    calib: object,
    uv: np.ndarray,
    depth: float,
    dimensions_lhw: np.ndarray,
) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape(2)
    center = calib.img_to_rect(
        np.asarray([uv[0]], dtype=np.float64),
        np.asarray([uv[1]], dtype=np.float64),
        np.asarray([depth], dtype=np.float64),
    )[0].astype(np.float64, copy=True)
    center[1] += float(dimensions_lhw[1]) / 2.0
    return center


def 应用组件替换(
    variant: str,
    dt: MutableMapping[str, np.ndarray],
    gt: Mapping[str, np.ndarray],
    dt_index: int,
    gt_index: int,
    calib: object,
) -> None:
    pred_location = np.asarray(dt["location"][dt_index], dtype=np.float64).copy()
    pred_dimensions = np.asarray(dt["dimensions"][dt_index], dtype=np.float64).copy()
    gt_location = np.asarray(gt["location"][gt_index], dtype=np.float64)
    gt_dimensions = np.asarray(gt["dimensions"][gt_index], dtype=np.float64)

    if variant == "GT轴向深度_保留预测射线":
        pred_uv = 投影中心(calib, pred_location, pred_dimensions)
        dt["location"][dt_index] = 从投影中心恢复底面位置(
            calib, pred_uv, float(gt_location[2]), pred_dimensions
        )
    elif variant == "GT投影中心_保留预测深度":
        gt_uv = 投影中心(calib, gt_location, gt_dimensions)
        dt["location"][dt_index] = 从投影中心恢复底面位置(
            calib, gt_uv, float(pred_location[2]), pred_dimensions
        )
    elif variant == "GT完整三维位置":
        dt["location"][dt_index] = gt_location
    elif variant == "GT三维尺寸":
        dt["dimensions"][dt_index] = gt_dimensions
    elif variant == "GT朝向":
        dt["rotation_y"][dt_index] = float(gt["rotation_y"][gt_index])
    elif variant == "GT全部三维属性":
        dt["location"][dt_index] = gt_location
        dt["dimensions"][dt_index] = gt_dimensions
        dt["rotation_y"][dt_index] = float(gt["rotation_y"][gt_index])
    else:
        raise KeyError(variant)

    if variant in {"GT朝向", "GT全部三维属性"}:
        bbox_center_x = float(np.mean(dt["bbox"][dt_index, [0, 2]]))
        dt["alpha"][dt_index] = float(
            calib.ry2alpha(float(dt["rotation_y"][dt_index]), bbox_center_x)
        )


def 深复制标注(annotation: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        key: np.asarray(value).copy()
        for key, value in annotation.items()
    }


def 审计全属性文本往返(
    gt_annos: Sequence[Mapping[str, np.ndarray]],
    roundtrip_annos: Sequence[Mapping[str, np.ndarray]],
    all_matches: Sequence[Sequence[Tuple[int, int, float]]],
) -> Dict[str, float]:
    """Verify that the anomalous full-attribute Oracle was written/read exactly.

    The previous G5 run produced an impossible-looking near-zero AP for the
    combined replacement.  This audit does not alter the metric: it checks the
    matched location, dimensions and yaw after KITTI text round-trip and stops
    immediately if serialization, dimension order or index pairing is wrong.
    """

    maximum_location_error = 0.0
    maximum_dimension_error = 0.0
    maximum_yaw_error = 0.0
    count = 0
    for gt, dt, matches in zip(gt_annos, roundtrip_annos, all_matches):
        for dt_index, gt_index, _ in matches:
            location_error = float(np.max(np.abs(
                np.asarray(dt["location"][dt_index], dtype=np.float64)
                - np.asarray(gt["location"][gt_index], dtype=np.float64)
            )))
            dimension_error = float(np.max(np.abs(
                np.asarray(dt["dimensions"][dt_index], dtype=np.float64)
                - np.asarray(gt["dimensions"][gt_index], dtype=np.float64)
            )))
            yaw_error = 角度差(
                float(dt["rotation_y"][dt_index]),
                float(gt["rotation_y"][gt_index]),
            )
            maximum_location_error = max(maximum_location_error, location_error)
            maximum_dimension_error = max(maximum_dimension_error, dimension_error)
            maximum_yaw_error = max(maximum_yaw_error, yaw_error)
            count += 1
    # G5 writes six decimal places.  Errors above two micro-units imply a real
    # indexing/order/serialization failure and make the Oracle invalid.
    tolerance = 2.0e-6
    if max(
        maximum_location_error,
        maximum_dimension_error,
        maximum_yaw_error,
    ) > tolerance:
        raise AssertionError(
            "G5全属性文本往返审计失败：location={}, dimensions={}, yaw={}".format(
                maximum_location_error,
                maximum_dimension_error,
                maximum_yaw_error,
            )
        )
    return {
        "匹配数": int(count),
        "最大位置往返误差": maximum_location_error,
        "最大尺寸往返误差": maximum_dimension_error,
        "最大朝向往返误差": maximum_yaw_error,
    }


def 保存标注目录(
    annotations: Sequence[Mapping[str, np.ndarray]],
    sample_ids: Sequence[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_id, anno in zip(sample_ids, annotations):
        path = output_dir / f"{sample_id}.txt"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            count = len(anno["name"])
            for index in range(count):
                dimensions = np.asarray(anno["dimensions"][index], dtype=np.float64)
                location = np.asarray(anno["location"][index], dtype=np.float64)
                bbox = np.asarray(anno["bbox"][index], dtype=np.float64)
                # 解析后的 [l,h,w] 恢复为 KITTI 文本中的 [h,w,l]。
                h, w, l = dimensions[1], dimensions[2], dimensions[0]
                score = float(np.asarray(anno["score"])[index])
                values = (
                    f"{anno['name'][index]} 0.0 0 "
                    f"{float(anno['alpha'][index]):.6f} "
                    f"{bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} "
                    f"{h:.6f} {w:.6f} {l:.6f} "
                    f"{location[0]:.6f} {location[1]:.6f} {location[2]:.6f} "
                    f"{float(anno['rotation_y'][index]):.6f} {score:.6f}\n"
                )
                stream.write(values)


def 评估结果目录(
    data_dir: Path,
    label_dir: Path,
    sample_ids: Sequence[str],
) -> Tuple[Dict[str, Dict[str, float]], List[Mapping[str, np.ndarray]], List[Mapping[str, np.ndarray]]]:
    from lib.datasets.kitti.kitti_eval_python import kitti_common as kitti
    from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result

    ids = [规范样本号(value) for value in sample_ids]
    dt_annos = kitti.get_label_annos(str(data_dir), ids)
    gt_annos = kitti.get_label_annos(str(label_dir), ids)
    metrics: Dict[str, Dict[str, float]] = {}
    for class_name, class_id in 类别编号.items():
        _, result_dict, _ = get_official_eval_result(gt_annos, dt_annos, class_id)
        metrics[class_name] = {
            "Easy": float(result_dict[f"{class_name}_3d_easy_R40"]),
            "Moderate": float(result_dict[f"{class_name}_3d_moderate_R40"]),
            "Hard": float(result_dict[f"{class_name}_3d_hard_R40"]),
        }
    return metrics, gt_annos, dt_annos


def 汇总匹配误差(
    gt_annos: Sequence[Mapping[str, np.ndarray]],
    dt_annos: Sequence[Mapping[str, np.ndarray]],
    all_matches: Sequence[Sequence[Tuple[int, int, float]]],
    calibs: Sequence[object],
) -> Dict[str, Dict[str, float]]:
    values: Dict[str, Dict[str, List[float]]] = {
        name: {
            "depth": [],
            "center": [],
            "dimension": [],
            "yaw": [],
            "iou2d": [],
        }
        for name in 类别编号
    }
    for gt, dt, matches, calib in zip(gt_annos, dt_annos, all_matches, calibs):
        for dt_index, gt_index, iou2d in matches:
            class_name = str(dt["name"][dt_index])
            pred_location = np.asarray(dt["location"][dt_index], dtype=np.float64)
            gt_location = np.asarray(gt["location"][gt_index], dtype=np.float64)
            pred_dims = np.asarray(dt["dimensions"][dt_index], dtype=np.float64)
            gt_dims = np.asarray(gt["dimensions"][gt_index], dtype=np.float64)
            center_error = np.linalg.norm(
                投影中心(calib, pred_location, pred_dims)
                - 投影中心(calib, gt_location, gt_dims)
            )
            relative_dim_error = np.mean(
                np.abs(pred_dims - gt_dims) / np.maximum(np.abs(gt_dims), 1.0e-6)
            )
            values[class_name]["depth"].append(abs(pred_location[2] - gt_location[2]))
            values[class_name]["center"].append(float(center_error))
            values[class_name]["dimension"].append(float(relative_dim_error))
            values[class_name]["yaw"].append(
                角度差(dt["rotation_y"][dt_index], gt["rotation_y"][gt_index])
            )
            values[class_name]["iou2d"].append(float(iou2d))

    summary: Dict[str, Dict[str, float]] = {}
    for class_name, class_values in values.items():
        count = len(class_values["depth"])
        summary[class_name] = {
            "匹配数": int(count),
            "深度MAE_m": float(np.mean(class_values["depth"])) if count else math.nan,
            "投影中心MAE_px": float(np.mean(class_values["center"])) if count else math.nan,
            "尺寸平均相对误差_pct": (
                100.0 * float(np.mean(class_values["dimension"])) if count else math.nan
            ),
            "朝向MAE_deg": (
                math.degrees(float(np.mean(class_values["yaw"]))) if count else math.nan
            ),
            "匹配2DIoU均值": float(np.mean(class_values["iou2d"])) if count else math.nan,
        }
    return summary


def 自动结论(
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> Tuple[str, str, Optional[str]]:
    baseline = float(metrics["V09原始预测"]["Car"]["Moderate"])
    mapping = {
        "GT轴向深度_保留预测射线": "MonoDGP式零初始化几何残差校正",
        "GT投影中心_保留预测深度": "可见区域约束的3D投影中心细化",
        "GT三维尺寸": "类别条件三维尺寸残差细化",
        "GT朝向": "周期分布式朝向细化",
    }
    gains = {
        variant: float(metrics[variant]["Car"]["Moderate"]) - baseline
        for variant in mapping
    }
    ordered = sorted(gains, key=gains.get, reverse=True)
    best = ordered[0]
    best_gain = gains[best]
    protected = all(
        float(metrics[best][name]["Moderate"])
        - float(metrics["V09原始预测"][name]["Moderate"])
        >= -0.10
        for name in 类别编号
    )
    if best_gain >= 0.50 and protected:
        return (
            "通过，允许实现短程配对版本",
            f"{best} 的 Car Moderate Oracle 提升 {best_gain:+.4f} AP，且三类 Moderate 均未触发保护线。",
            mapping[best],
        )
    if best_gain >= 0.30:
        return (
            "谨慎，仅允许先做更小规模验证",
            f"{best} 的 Car Moderate Oracle 提升 {best_gain:+.4f} AP，但收益或多类别保护尚不充分。",
            mapping[best],
        )
    full_location_gain = (
        float(metrics["GT完整三维位置"]["Car"]["Moderate"]) - baseline
    )
    if full_location_gain >= 0.50:
        return (
            "单组件否决，位置交互仍有上限",
            "单一组件均不足，但完整位置 Oracle 有明显空间；先分析深度与投影中心耦合，不实现正式模块。",
            None,
        )
    return (
        "否决，不新增三维回归模块",
        "各单组件 Oracle 均未达到 0.30 AP，现有候选的主要问题不在单一三维属性。",
        None,
    )


def 写报告(
    path: Path,
    results_dir: Path,
    match_threshold: float,
    matched_count: int,
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    error_summary: Mapping[str, Mapping[str, float]],
    conclusion: Tuple[str, str, Optional[str]],
    elapsed_seconds: float,
) -> None:
    baseline = metrics["V09原始预测"]
    lines = [
        "# StereoDETR V09三维框组件误差归因G5报告",
        "",
        "## 结论先行",
        "",
        f"- 自动结论：**{conclusion[0]}**。{conclusion[1]}",
        f"- 下一候选：**{conclusion[2] or '无，继续误差归因'}**。",
        "- 本报告中的 GT 替换均使用验证集真值，只是诊断 Oracle，不能写成模型精度或提交 KITTI。",
        "- 匹配只使用类别与 2D IoU，没有使用深度、位置、尺寸或朝向，因此不会偏向某个被诊断的 3D 组件。",
        "",
        "## 诊断设置",
        "",
        f"- 原始预测目录：`{results_dir}`",
        f"- 一对一匹配最低 2D IoU：`{match_threshold:.2f}`",
        f"- 成功匹配数：`{matched_count}`",
        f"- 总耗时：`{elapsed_seconds / 60.0:.2f}` 分钟",
        "",
        "## KITTI验证集3D AP_R40组件Oracle",
        "",
        "| 变体 | Car E/M/H | ΔCar M | Pedestrian E/M/H | ΔPed M | Cyclist E/M/H | ΔCyc M |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, per_class in metrics.items():
        delta_car = per_class["Car"]["Moderate"] - baseline["Car"]["Moderate"]
        delta_ped = per_class["Pedestrian"]["Moderate"] - baseline["Pedestrian"]["Moderate"]
        delta_cyc = per_class["Cyclist"]["Moderate"] - baseline["Cyclist"]["Moderate"]
        def triplet(name: str) -> str:
            item = per_class[name]
            return f"{item['Easy']:.4f}/{item['Moderate']:.4f}/{item['Hard']:.4f}"
        lines.append(
            f"| {variant} | {triplet('Car')} | {delta_car:+.4f} | "
            f"{triplet('Pedestrian')} | {delta_ped:+.4f} | "
            f"{triplet('Cyclist')} | {delta_cyc:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 一对一匹配后的连续误差",
            "",
            "| 类别 | 匹配数 | 深度MAE(m) | 投影中心MAE(px) | 尺寸相对误差 | 朝向MAE(°) | 匹配2D IoU |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for class_name, item in error_summary.items():
        lines.append(
            f"| {class_name} | {int(item['匹配数'])} | {item['深度MAE_m']:.4f} | "
            f"{item['投影中心MAE_px']:.4f} | {item['尺寸平均相对误差_pct']:.2f}% | "
            f"{item['朝向MAE_deg']:.2f} | {item['匹配2DIoU均值']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 如何解释各变体",
            "",
            "- `GT轴向深度_保留预测射线`：只修正沿相机光轴的距离，并保持模型预测的3D投影中心；最贴近轻量深度残差头的能力边界。",
            "- `GT投影中心_保留预测深度`：只修正3D中心在图像上的投影位置；用于判断可见区域采样和中心回归是否仍是瓶颈。",
            "- `GT完整三维位置`：同时修正深度与投影中心，衡量二者耦合后的平移上限。",
            "- `GT三维尺寸`与`GT朝向`：分别隔离尺寸和旋转误差。",
            "- `GT全部三维属性`：是候选检测与评分固定时的3D框回归上限，不是可实现模型结果。",
            "",
            "## 预注册决策线",
            "",
            "1. 单组件 Car Moderate Oracle 至少 `+0.50 AP`，且三类 Moderate 均不低于原始值 `0.10 AP` 以上，才允许实现正式短程配对版本。",
            "2. `+0.30~+0.49 AP` 只允许做更小规模、零初始化、冻结主干的验证，不跑195轮。",
            "3. 低于 `+0.30 AP` 直接否决对应模块；不得因为论文新或模块热门而继续租卡。",
            "4. 通过 G5 后仍需同起点、同种子控制组和新增模块组配对，并做第二随机种子与同步CUDA时延复测。",
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
    import tempfile

    from lib.datasets.kitti.kitti_utils import Calibration

    boxes = np.asarray([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64)
    iou = 二维IoU矩阵(boxes, boxes)
    assert np.allclose(iou, np.eye(2))
    assert abs(角度差(math.pi - 0.1, -math.pi + 0.1) - 0.2) < 1.0e-8

    calib = Calibration(
        {
            "P2": np.asarray([[100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
            "P3": np.asarray([[100, 0, 0, -50], [0, 100, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
            "R0": np.eye(3, dtype=np.float32),
            "Tr_velo2cam": np.concatenate(
                [np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)],
                axis=1,
            ),
        }
    )
    pred_dims = np.asarray([4.0, 2.0, 2.0])
    pred_loc = np.asarray([2.0, 3.0, 10.0])
    uv = 投影中心(calib, pred_loc, pred_dims)
    rebuilt = 从投影中心恢复底面位置(calib, uv, 20.0, pred_dims)
    assert np.allclose(rebuilt, np.asarray([4.0, 5.0, 20.0]), atol=1.0e-6)

    gt = {
        "name": np.asarray(["Car"]),
        "bbox": np.asarray([[0.0, 0.0, 10.0, 10.0]]),
    }
    dt = {
        "name": np.asarray(["Car"]),
        "bbox": np.asarray([[1.0, 1.0, 9.0, 9.0]]),
    }
    matches = 一对一二维匹配(gt, dt, 0.3)
    assert len(matches) == 1 and matches[0][:2] == (0, 0)

    annotation = {
        "name": np.asarray(["Car"]),
        "alpha": np.asarray([0.25]),
        "bbox": np.asarray([[10.0, 20.0, 30.0, 40.0]]),
        "dimensions": np.asarray([[4.2, 1.6, 1.8]]),
        "location": np.asarray([[1.0, 2.0, 20.0]]),
        "rotation_y": np.asarray([0.3]),
        "score": np.asarray([0.9]),
    }
    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir = Path(temporary_dir)
        保存标注目录([annotation], ["000001"], output_dir)
        from lib.datasets.kitti.kitti_eval_python import kitti_common as kitti

        loaded = kitti.get_label_annos(str(output_dir), ["000001"])[0]
        assert loaded["name"].tolist() == ["Car"]
        for key in ("dimensions", "location", "rotation_y", "score"):
            assert np.allclose(loaded[key], annotation[key])
        audit = 审计全属性文本往返(
            [annotation], [loaded], [[(0, 0, 1.0)]]
        )
        assert audit["匹配数"] == 1

    print(
        "G5自检通过：2D匹配、角度环绕、投影中心、沿射线深度替换、"
        "KITTI文本读回与全属性往返审计正常"
    )


def 主程序(args: argparse.Namespace) -> None:
    from lib.datasets.kitti.kitti_utils import Calibration
    from lib.helpers.config_helper import load_config

    if not 0.0 <= args.match_iou_2d <= 1.0:
        raise ValueError("match_iou_2d 必须位于 [0,1]")
    start = time.time()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["trainer"]

    results_dir = Path(args.results_dir) if args.results_dir else Path(
        train_cfg["save_path"]
    ) / cfg["model_name"] / "outputs" / "data"
    label_dir = Path(args.label_dir) if args.label_dir else Path(
        dataset_cfg["root_dir_eval"]
    ) / "label_2"
    calib_dir = Path(args.calib_dir) if args.calib_dir else Path(
        dataset_cfg["root_dir_eval"]
    ) / "calib"
    split_file = Path(args.split_file) if args.split_file else Path(
        dataset_cfg["eval_txt"]
    )
    output_dir = Path(args.output_dir)

    for path, name in (
        (results_dir, "V09预测目录"),
        (label_dir, "标签目录"),
        (calib_dir, "标定目录"),
        (split_file, "验证划分"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"找不到{name}：{path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = 读取样本号(split_file)
    baseline_metrics, gt_annos, dt_annos = 评估结果目录(
        results_dir, label_dir, sample_ids
    )
    calibs = [Calibration(str(calib_dir / f"{sample_id}.txt")) for sample_id in sample_ids]
    all_matches = [
        一对一二维匹配(gt, dt, args.match_iou_2d)
        for gt, dt in zip(gt_annos, dt_annos)
    ]
    matched_count = sum(len(item) for item in all_matches)
    if matched_count < 100:
        raise RuntimeError(
            f"仅匹配到 {matched_count} 个对象，路径或阈值可能错误，停止 Oracle 评估"
        )

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        "V09原始预测": baseline_metrics
    }
    full_attribute_audit = None
    for variant in 组件变体:
        repaired_annos: List[Dict[str, np.ndarray]] = []
        for gt, dt, matches, calib in zip(gt_annos, dt_annos, all_matches, calibs):
            repaired = 深复制标注(dt)
            for dt_index, gt_index, _ in matches:
                应用组件替换(variant, repaired, gt, dt_index, gt_index, calib)
            repaired_annos.append(repaired)
        variant_data = output_dir / "Oracle_仅诊断禁止提交" / variant / "data"
        保存标注目录(repaired_annos, sample_ids, variant_data)
        metrics[variant], _, roundtrip_annos = 评估结果目录(
            variant_data, label_dir, sample_ids
        )
        if variant == "GT全部三维属性":
            full_attribute_audit = 审计全属性文本往返(
                gt_annos, roundtrip_annos, all_matches
            )

    error_summary = 汇总匹配误差(
        gt_annos, dt_annos, all_matches, calibs
    )
    conclusion = 自动结论(metrics)
    summary = {
        "原始预测目录": str(results_dir),
        "样本数": len(sample_ids),
        "匹配阈值": args.match_iou_2d,
        "匹配数": matched_count,
        "指标": metrics,
        "连续误差": error_summary,
        "全属性文本往返审计": full_attribute_audit,
        "自动结论": {
            "等级": conclusion[0],
            "说明": conclusion[1],
            "下一候选": conclusion[2],
        },
        "限制": "所有GT替换结果均为验证集Oracle，只用于诊断，禁止作为模型精度。",
    }
    (output_dir / "诊断汇总.json").write_text(
        json.dumps(转JSON安全(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "诊断报告.md"
    写报告(
        report_path,
        results_dir,
        args.match_iou_2d,
        matched_count,
        metrics,
        error_summary,
        conclusion,
        time.time() - start,
    )
    print("G5诊断完成")
    print("报告:", report_path)
    print("自动结论:", conclusion[0])
    print("下一候选:", conclusion[2] or "无")


def main() -> None:
    args = 解析参数()
    if args.self_test:
        自检()
        return
    主程序(args)


if __name__ == "__main__":
    main()
