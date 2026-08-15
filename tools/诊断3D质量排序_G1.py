#!/usr/bin/env python3
"""G1：诊断 StereoDETR 的检测分数是否与真实 3D 定位质量对齐。

本脚本只进行一次验证集推理，不训练、不改权重。它完成三件事：

1. 分别统计分类分数、深度置信度和当前乘积分数与同类 GT 最大 3D IoU 的
   Pearson/Spearman 相关性；
2. 扫描 ``class_score * depth_confidence ** gamma``，判断现有 sigma 是否只需
   重新校准；
3. 用真实 3D IoU 作为仅供诊断的 Oracle 分数，估计学习 3D 质量排序的上限。

Oracle 使用了验证集 GT，绝不能作为正式模型结果或提交结果。它只用于决定是否
值得实现后续的 V06B/V06C。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


项目根目录 = Path(__file__).resolve().parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))


类别编号 = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
}


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G1：StereoDETR 评分与真实 3D IoU 对齐诊断"
    )
    parser.add_argument(
        "--config",
        default="versions/V00_官方基线/config.yaml",
        help="V00 配置文件",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="V00 checkpoint_best.pth；省略时从配置自动推导",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/三维质量排序诊断_G1",
    )
    parser.add_argument(
        "--gammas",
        default="0,0.25,0.5,0.75,1,1.5,2",
        help="sigma 指数列表；gamma=1 是当前 StereoDETR 分数",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="仅用于冒烟测试；0 表示完整验证集",
    )
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def 解析指数(文本: str) -> List[float]:
    结果 = []
    for 分段 in 文本.split(","):
        分段 = 分段.strip()
        if not 分段:
            continue
        数值 = float(分段)
        if not math.isfinite(数值) or 数值 < 0:
            raise ValueError(f"gamma 必须是非负有限数：{分段}")
        if 数值 not in 结果:
            结果.append(数值)
    if 1.0 not in 结果:
        结果.append(1.0)
    return sorted(结果)


def 指数目录名(gamma: float) -> str:
    文本 = f"{gamma:g}".replace(".", "p")
    return f"gamma_{文本}"


def 规范样本号(value: object) -> str:
    """保持本项目 KITTI 读取器要求的六位字符串编号。"""

    文本 = str(value)
    return f"{int(文本):06d}" if 文本.isdigit() else 文本


def 自动权重路径(cfg: Mapping[str, object]) -> Path:
    trainer = cfg["trainer"]
    assert isinstance(trainer, Mapping)
    return (
        项目根目录
        / str(trainer["save_path"])
        / str(cfg["model_name"])
        / "checkpoint_best.pth"
    )


def 复制并设置分数(
    几何结果: Mapping[str, Sequence[Sequence[float]]],
    元数据: Mapping[str, Sequence[Mapping[str, float]]],
    gamma: float,
) -> Dict[str, List[List[float]]]:
    结果: Dict[str, List[List[float]]] = {}
    for 样本号, 检测列表 in 几何结果.items():
        当前元数据 = 元数据[样本号]
        if len(检测列表) != len(当前元数据):
            raise AssertionError(
                f"{样本号} 解码检测数与评分元数据不一致："
                f"{len(检测列表)} != {len(当前元数据)}"
            )
        当前结果 = []
        for 检测, 评分 in zip(检测列表, 当前元数据):
            新检测 = list(检测)
            分类分数 = float(评分["class_score"])
            深度置信度 = max(float(评分["depth_confidence"]), 1.0e-12)
            新检测[-1] = 分类分数 * 深度置信度 ** gamma
            当前结果.append(新检测)
        结果[样本号] = 当前结果
    return 结果


def 保存KITTI结果(
    结果: Mapping[str, Sequence[Sequence[float]]],
    输出目录: Path,
    类别名称: Sequence[str],
    分数精度: int = 2,
) -> None:
    输出目录.mkdir(parents=True, exist_ok=True)
    for 样本号, 检测列表 in 结果.items():
        路径 = 输出目录 / f"{样本号}.txt"
        with 路径.open("w", encoding="utf-8", newline="\n") as stream:
            for 检测 in 检测列表:
                类别 = 类别名称[int(检测[0])]
                # 几何值和普通评分沿用官方 Tester 的两位小数写法，以便
                # gamma=1 尽可能复现既有 V00 指标。Oracle 单独提高分数
                # 精度，避免真实 IoU 排序因两位小数产生大量并列。
                几何数值 = [f"{float(value):.2f}" for value in 检测[1:-1]]
                分数字符串 = f"{float(检测[-1]):.{分数精度}f}"
                数值 = " ".join([*几何数值, 分数字符串])
                stream.write(f"{类别} 0.0 0 {数值}\n")


def 三维框数组(标注: Mapping[str, np.ndarray], 索引: np.ndarray) -> np.ndarray:
    if not int(np.asarray(索引).size):
        return np.empty((0, 7), dtype=np.float64)
    return np.concatenate(
        [
            np.asarray(标注["location"])[索引],
            np.asarray(标注["dimensions"])[索引],
            np.asarray(标注["rotation_y"])[索引, None],
        ],
        axis=1,
    ).astype(np.float64, copy=False)


def 计算同类最大三维IoU(
    gt_annos: Sequence[Mapping[str, np.ndarray]],
    dt_annos: Sequence[Mapping[str, np.ndarray]],
) -> List[np.ndarray]:
    """使用项目自带 KITTI 3D IoU，返回每个检测对同类 GT 的最大 IoU。"""

    from lib.datasets.kitti.kitti_eval_python.eval import d3_box_overlap

    所有质量: List[np.ndarray] = []
    for gt, dt in zip(gt_annos, dt_annos):
        质量 = np.zeros(len(dt["name"]), dtype=np.float64)
        for 类别 in 类别编号:
            gt索引 = np.flatnonzero(np.asarray(gt["name"]) == 类别)
            dt索引 = np.flatnonzero(np.asarray(dt["name"]) == 类别)
            if not len(gt索引) or not len(dt索引):
                continue
            gt框 = 三维框数组(gt, gt索引)
            dt框 = 三维框数组(dt, dt索引)
            iou = d3_box_overlap(gt框, dt框)
            质量[dt索引] = np.max(iou, axis=0)
        所有质量.append(np.clip(质量, 0.0, 1.0))
    return 所有质量


def 安全相关性(x: Iterable[float], y: Iterable[float]) -> Dict[str, object]:
    from scipy.stats import pearsonr, spearmanr

    x数组 = np.asarray(list(x), dtype=np.float64)
    y数组 = np.asarray(list(y), dtype=np.float64)
    有效 = np.isfinite(x数组) & np.isfinite(y数组)
    x数组 = x数组[有效]
    y数组 = y数组[有效]
    if len(x数组) < 3 or np.std(x数组) < 1.0e-12 or np.std(y数组) < 1.0e-12:
        return {"数量": int(len(x数组)), "Pearson": None, "Spearman": None}
    return {
        "数量": int(len(x数组)),
        "Pearson": float(pearsonr(x数组, y数组)[0]),
        "Spearman": float(spearmanr(x数组, y数组)[0]),
    }


def 评估结果目录(
    数据目录: Path,
    标签目录: Path,
    样本号: Sequence[str],
) -> Tuple[Dict[str, Dict[str, float]], List[Mapping[str, np.ndarray]], List[Mapping[str, np.ndarray]]]:
    from lib.datasets.kitti.kitti_eval_python import kitti_common as kitti
    from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result

    # 当前工程的 get_image_index_str() 原样返回输入值，随后直接拼接
    # '.txt'，因此必须传入 '000123'，不能传入整数 123。
    字符串样本号 = [规范样本号(value) for value in 样本号]
    dt_annos = kitti.get_label_annos(str(数据目录), 字符串样本号)
    gt_annos = kitti.get_label_annos(str(标签目录), 字符串样本号)
    指标: Dict[str, Dict[str, float]] = {}
    for 类别, 编号 in 类别编号.items():
        _, 结果字典, _ = get_official_eval_result(gt_annos, dt_annos, 编号)
        指标[类别] = {
            "3D_AP_R40_Easy": float(结果字典[f"{类别}_3d_easy_R40"]),
            "3D_AP_R40_Moderate": float(结果字典[f"{类别}_3d_moderate_R40"]),
            "3D_AP_R40_Hard": float(结果字典[f"{类别}_3d_hard_R40"]),
            "BEV_AP_R40_Moderate": float(结果字典[f"{类别}_bev_moderate_R40"]),
        }
    return 指标, gt_annos, dt_annos


def 构建检测记录(
    样本号: Sequence[str],
    dt_annos: Sequence[Mapping[str, np.ndarray]],
    元数据: Mapping[str, Sequence[Mapping[str, float]]],
    三维质量: Sequence[np.ndarray],
) -> List[Dict[str, object]]:
    记录: List[Dict[str, object]] = []
    for 当前样本号, dt, 当前质量 in zip(样本号, dt_annos, 三维质量):
        当前元数据 = 元数据[当前样本号]
        if len(dt["name"]) != len(当前元数据) or len(当前质量) != len(当前元数据):
            raise AssertionError(f"{当前样本号} 的检测、元数据和 IoU 数量不一致")
        for 索引, (类别, 评分, iou) in enumerate(
            zip(dt["name"], 当前元数据, 当前质量)
        ):
            分类分数 = float(评分["class_score"])
            深度置信度 = float(评分["depth_confidence"])
            记录.append(
                {
                    "样本号": 当前样本号,
                    "检测序号": 索引,
                    "类别": str(类别),
                    "分类分数": 分类分数,
                    "深度置信度": 深度置信度,
                    "当前乘积分数": 分类分数 * 深度置信度,
                    "同类GT最大3DIoU": float(iou),
                }
            )
    return 记录


def 汇总相关性(记录: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, Dict[str, object]]]:
    分组: MutableMapping[str, List[Mapping[str, object]]] = {"全部": list(记录)}
    for 类别 in 类别编号:
        分组[类别] = [item for item in 记录 if item["类别"] == 类别]
    分组["存在同类重叠"] = [
        item for item in 记录 if float(item["同类GT最大3DIoU"]) > 0.01
    ]

    结果: Dict[str, Dict[str, Dict[str, object]]] = {}
    for 组名, 当前记录 in 分组.items():
        质量 = [float(item["同类GT最大3DIoU"]) for item in 当前记录]
        结果[组名] = {
            "分类分数": 安全相关性(
                [float(item["分类分数"]) for item in 当前记录], 质量
            ),
            "深度置信度": 安全相关性(
                [float(item["深度置信度"]) for item in 当前记录], 质量
            ),
            "当前乘积分数": 安全相关性(
                [float(item["当前乘积分数"]) for item in 当前记录], 质量
            ),
        }
    return 结果


def 构建Oracle结果(
    当前结果: Mapping[str, Sequence[Sequence[float]]],
    样本号: Sequence[str],
    三维质量: Sequence[np.ndarray],
) -> Dict[str, List[List[float]]]:
    结果: Dict[str, List[List[float]]] = {}
    for 当前样本号, 当前质量 in zip(样本号, 三维质量):
        检测列表 = 当前结果[当前样本号]
        if len(检测列表) != len(当前质量):
            raise AssertionError(f"{当前样本号} Oracle 质量数量不一致")
        当前列表 = []
        for 检测, iou in zip(检测列表, 当前质量):
            新检测 = list(检测)
            # 原分数只负责相同 IoU 时的稳定排序，不改变主要 Oracle 顺序。
            原分数 = float(新检测[-1])
            新检测[-1] = float(iou) + 1.0e-4 * float(np.clip(原分数, 0.0, 1.0))
            当前列表.append(新检测)
        结果[当前样本号] = 当前列表
    return 结果


def 自动结论(当前值: float, 最佳值: float, oracle值: float) -> Tuple[str, str]:
    校准提升 = 最佳值 - 当前值
    oracle提升 = oracle值 - 当前值
    if oracle提升 < 0.3:
        return (
            "否决",
            "Oracle 提升不足 0.3 AP；现有候选的排序空间太小，不进入 V06。",
        )
    if 校准提升 >= 0.3:
        return (
            "先做零训练校准",
            "现有 sigma 指数扫描已提升至少 0.3 AP；先复核最佳 gamma，再决定是否训练质量损失。",
        )
    if oracle提升 >= 0.8:
        return (
            "通过",
            "现有 sigma 校准不足，但 Oracle 至少提升 0.8 AP；值得实现学习式 3D 质量排序。",
        )
    return (
        "谨慎",
        "Oracle 提升位于 0.3–0.8 AP；只能先做同起点短程配对实验。",
    )


def 写CSV(路径: Path, 记录: Sequence[Mapping[str, object]]) -> None:
    路径.parent.mkdir(parents=True, exist_ok=True)
    字段 = [
        "样本号",
        "检测序号",
        "类别",
        "分类分数",
        "深度置信度",
        "当前乘积分数",
        "同类GT最大3DIoU",
    ]
    with 路径.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=字段)
        writer.writeheader()
        writer.writerows(记录)


def 格式化数值(value: object, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "N/A"


def 写报告(
    路径: Path,
    权重路径: Path,
    样本数: int,
    gamma指标: Mapping[float, Mapping[str, Mapping[str, float]]],
    oracle指标: Mapping[str, Mapping[str, float]],
    相关性: Mapping[str, Mapping[str, Mapping[str, object]]],
    结论: Tuple[str, str],
    耗时秒: float,
) -> None:
    当前 = gamma指标[1.0]["Car"]["3D_AP_R40_Moderate"]
    最佳gamma = max(
        gamma指标,
        key=lambda item: gamma指标[item]["Car"]["3D_AP_R40_Moderate"],
    )
    最佳 = gamma指标[最佳gamma]["Car"]["3D_AP_R40_Moderate"]
    oracle = oracle指标["Car"]["3D_AP_R40_Moderate"]

    行 = [
        "# StereoDETR 三维质量排序 G1 诊断报告",
        "",
        "## 结论先行",
        "",
        f"- 自动结论：**{结论[0]}**。{结论[1]}",
        f"- 当前 gamma=1：Car 3D AP_R40 Moderate = **{当前:.4f}**。",
        f"- 最佳 gamma={最佳gamma:g}：**{最佳:.4f}**，变化 **{最佳-当前:+.4f}**。",
        f"- 3D IoU Oracle：**{oracle:.4f}**，相对当前 **{oracle-当前:+.4f}**。",
        "- Oracle 使用验证集真值，只是诊断上限，不能作为模型结果或论文实验结果。",
        "",
        "## 实验信息",
        "",
        f"- 权重：`{权重路径}`",
        f"- 验证样本：{样本数}",
        f"- 总耗时：{耗时秒/60.0:.2f} 分钟",
        "- 当前 StereoDETR 分数定义：`class_score × depth_confidence`，即 gamma=1。",
        "- Top-K 仍由分类分数预选；本诊断只重新排列已经进入 Top-K 的检测。",
        "",
        "## sigma 指数扫描",
        "",
        "| gamma | Car Easy | Car Moderate | Car Hard | 相对当前 Moderate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for gamma in sorted(gamma指标):
        car = gamma指标[gamma]["Car"]
        moderate = car["3D_AP_R40_Moderate"]
        行.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:+.4f} |".format(
                f"{gamma:g}",
                car["3D_AP_R40_Easy"],
                moderate,
                car["3D_AP_R40_Hard"],
                moderate - 当前,
            )
        )

    行.extend(
        [
            "",
            "## 分数与真实 3D IoU 的相关性",
            "",
            "| 分组 | 分数 | 数量 | Pearson | Spearman |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for 组名, 分数结果 in 相关性.items():
        for 分数名, 数值 in 分数结果.items():
            行.append(
                "| {} | {} | {} | {} | {} |".format(
                    组名,
                    分数名,
                    数值["数量"],
                    格式化数值(数值["Pearson"]),
                    格式化数值(数值["Spearman"]),
                )
            )

    行.extend(
        [
            "",
            "## Oracle 对照",
            "",
            "| 类别 | 当前 Moderate | Oracle Moderate | 提升 |",
            "|---|---:|---:|---:|",
        ]
    )
    for 类别 in 类别编号:
        当前类别 = gamma指标[1.0][类别]["3D_AP_R40_Moderate"]
        oracle类别 = oracle指标[类别]["3D_AP_R40_Moderate"]
        行.append(
            f"| {类别} | {当前类别:.4f} | {oracle类别:.4f} | "
            f"{oracle类别-当前类别:+.4f} |"
        )

    行.extend(
        [
            "",
            "## 如何决定下一步",
            "",
            "1. Oracle 提升小于 0.3：停止质量排序路线。",
            "2. 最佳 gamma 已提升至少 0.3：先固定 gamma 做复核，不训练新模块。",
            "3. gamma 无明显收益、Oracle 提升至少 0.8：实现 V06B 的真实 3D IoU 软目标。",
            "4. V06B 短程配对有效后，再加入 V06C 的 pair-wise 排序损失。",
            "",
            "原始逐检测记录见 `检测质量明细.csv`，机器可读汇总见 `诊断汇总.json`。",
        ]
    )
    路径.write_text("\n".join(行) + "\n", encoding="utf-8")


def 转为JSON安全(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): 转为JSON安全(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [转为JSON安全(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def 自检() -> None:
    assert 解析指数("1,0,0.5,1") == [0.0, 0.5, 1.0]
    assert 规范样本号(123) == "000123"
    assert 规范样本号("000123") == "000123"
    几何 = {"000001": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.0]]}
    元数据 = {
        "000001": [{"class_score": 0.8, "depth_confidence": 0.5}]
    }
    assert abs(复制并设置分数(几何, 元数据, 0.0)["000001"][0][-1] - 0.8) < 1e-8
    assert abs(复制并设置分数(几何, 元数据, 1.0)["000001"][0][-1] - 0.4) < 1e-8
    assert 自动结论(48.0, 48.1, 48.2)[0] == "否决"
    assert 自动结论(48.0, 48.4, 49.0)[0] == "先做零训练校准"
    assert 自动结论(48.0, 48.1, 49.0)[0] == "通过"
    相关 = 安全相关性([0, 1, 2, 3], [0, 1, 2, 3])
    assert abs(float(相关["Pearson"]) - 1.0) < 1e-8
    assert abs(float(相关["Spearman"]) - 1.0) < 1e-8
    print("G1 自检通过：指数评分、止损门槛和相关性计算正常")


def 主程序(args: argparse.Namespace) -> None:
    import torch
    import tqdm

    from lib.helpers.config_helper import load_config
    from lib.helpers.decode_helper import decode_detections, extract_dets_from_outputs
    from lib.helpers.qecr_dataloader_helper import build_qecr_dataloader
    from lib.helpers.qecr_model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("完整 G1 需要 CUDA：模型推理和项目自带 3D IoU 都使用 GPU")

    开始 = time.time()
    cfg = load_config(args.config)
    gamma列表 = 解析指数(args.gammas)
    输出根目录 = Path(args.output_dir).expanduser().resolve()
    输出根目录.mkdir(parents=True, exist_ok=True)
    权重路径 = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else 自动权重路径(cfg).resolve()
    )
    if not 权重路径.is_file():
        raise FileNotFoundError(f"找不到 V00 权重：{权重路径}")

    workers = int(cfg["dataset"].get("workers", 8))
    _, 验证加载器 = build_qecr_dataloader(cfg["dataset"], workers=workers)
    model, _ = build_model(cfg["model"])
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    logger = logging.getLogger("StereoDETR_G1")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    load_checkpoint(
        model=model,
        optimizer=None,
        filename=str(权重路径),
        map_location=device,
        logger=logger,
    )

    几何结果: Dict[str, List[List[float]]] = {}
    元数据: Dict[str, List[Dict[str, float]]] = {}
    threshold = float(cfg["tester"].get("threshold", 0.2))
    topk = int(cfg["tester"]["topk"])
    max_objs = int(验证加载器.dataset.max_objs)

    progress = tqdm.tqdm(total=len(验证加载器), desc="G1 inference")
    with torch.no_grad():
        for batch_index, (inputs, calibs, targets, info) in enumerate(验证加载器):
            inputs = inputs.to(device)
            calibs_tensor = calibs.to(device)
            img_sizes = info["img_size_croped"].to(device)
            img_sizes_ori = info["img_size_original"].to(device)
            img_sizes_upper = info["upper"].to(device)
            outputs = model(
                inputs,
                calibs_tensor,
                targets,
                img_sizes,
                img_sizes_ori,
                img_sizes_upper,
                dn_args=0,
            )
            dets = extract_dets_from_outputs(outputs=outputs, K=max_objs, topk=topk)
            dets = dets.detach().cpu().numpy()

            原始info = info
            标定列表 = [
                验证加载器.dataset.get_calib(index) for index in 原始info["img_id"]
            ]
            numpy_info = {
                key: value.detach().cpu().numpy() if key != "img_id" else value
                for key, value in 原始info.items()
            }
            当前结果, _ = decode_detections(
                dets=dets.copy(),
                info=numpy_info,
                calibs=标定列表,
                cls_mean_size=验证加载器.dataset.cls_mean_size,
                threshold=threshold,
                decoupled=cfg["tester"].get("decoupled", False),
            )
            for 图内索引, 原样本号 in enumerate(原始info["img_id"]):
                样本号 = 规范样本号(原样本号)
                行 = dets[图内索引]
                有效行 = 行[行[:, 1] >= threshold]
                当前元数据 = [
                    {
                        "class_score": float(item[1]),
                        "depth_confidence": float(item[-1]),
                    }
                    for item in 有效行
                ]
                # decode_detections 沿用原始 img_id 作为键。
                原键 = 原样本号
                解码列表 = 当前结果[原键]
                几何结果[样本号] = [list(item) for item in 解码列表]
                元数据[样本号] = 当前元数据
                if len(几何结果[样本号]) != len(当前元数据):
                    raise AssertionError(f"{样本号} 解码后检测数量不一致")

            progress.update(1)
            if args.max_batches > 0 and batch_index + 1 >= args.max_batches:
                break
    progress.close()

    样本号列表 = sorted(几何结果)
    类别名称 = list(验证加载器.dataset.writelist_train)
    标签目录 = Path(验证加载器.dataset.label_dir)
    gamma指标: Dict[float, Dict[str, Dict[str, float]]] = {}
    gamma结果: Dict[float, Dict[str, List[List[float]]]] = {}
    gamma1_gt = None
    gamma1_dt = None
    for gamma in gamma列表:
        当前结果 = 复制并设置分数(几何结果, 元数据, gamma)
        gamma结果[gamma] = 当前结果
        数据目录 = 输出根目录 / "指数扫描" / 指数目录名(gamma) / "data"
        保存KITTI结果(当前结果, 数据目录, 类别名称)
        指标, gt_annos, dt_annos = 评估结果目录(
            数据目录, 标签目录, 样本号列表
        )
        gamma指标[gamma] = 指标
        if gamma == 1.0:
            gamma1_gt = gt_annos
            gamma1_dt = dt_annos

    assert gamma1_gt is not None and gamma1_dt is not None
    三维质量 = 计算同类最大三维IoU(gamma1_gt, gamma1_dt)
    检测记录 = 构建检测记录(样本号列表, gamma1_dt, 元数据, 三维质量)
    相关性 = 汇总相关性(检测记录)

    oracle结果 = 构建Oracle结果(gamma结果[1.0], 样本号列表, 三维质量)
    oracle目录 = 输出根目录 / "Oracle_仅诊断禁止提交" / "data"
    保存KITTI结果(oracle结果, oracle目录, 类别名称, 分数精度=6)
    oracle指标, _, _ = 评估结果目录(oracle目录, 标签目录, 样本号列表)

    当前值 = gamma指标[1.0]["Car"]["3D_AP_R40_Moderate"]
    最佳gamma = max(
        gamma指标,
        key=lambda item: gamma指标[item]["Car"]["3D_AP_R40_Moderate"],
    )
    最佳值 = gamma指标[最佳gamma]["Car"]["3D_AP_R40_Moderate"]
    oracle值 = oracle指标["Car"]["3D_AP_R40_Moderate"]
    结论 = 自动结论(当前值, 最佳值, oracle值)

    写CSV(输出根目录 / "检测质量明细.csv", 检测记录)
    汇总 = {
        "权重": str(权重路径),
        "样本数": len(样本号列表),
        "gamma指标": gamma指标,
        "oracle指标": oracle指标,
        "相关性": 相关性,
        "自动结论": {"等级": 结论[0], "说明": 结论[1]},
        "限制": "Oracle 使用验证集 GT，仅用于诊断；Top-K 仍由分类分数预选。",
    }
    (输出根目录 / "诊断汇总.json").write_text(
        json.dumps(转为JSON安全(汇总), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    写报告(
        输出根目录 / "诊断报告.md",
        权重路径,
        len(样本号列表),
        gamma指标,
        oracle指标,
        相关性,
        结论,
        time.time() - 开始,
    )

    print("\nG1 诊断完成")
    print("报告:", 输出根目录 / "诊断报告.md")
    print("当前 Car 3D AP_R40 Moderate:", f"{当前值:.4f}")
    print("最佳 gamma:", f"{最佳gamma:g}", "指标:", f"{最佳值:.4f}")
    print("Oracle 指标:", f"{oracle值:.4f}")
    print("自动结论:", 结论[0], "-", 结论[1])


def main() -> None:
    args = 解析参数()
    if args.self_test:
        自检()
        return
    主程序(args)


if __name__ == "__main__":
    main()
