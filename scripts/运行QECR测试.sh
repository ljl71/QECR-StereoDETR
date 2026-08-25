#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

bash -n scripts/运行V08短程实验.sh
bash -n scripts/运行V09强基线质量组合.sh
bash -n scripts/运行V10前景TopK深度诊断.sh
bash -n scripts/运行V11动态深度上采样.sh
bash -n scripts/运行V12分组相关诊断.sh
bash -n scripts/运行V12轻量分组相关.sh
bash -n scripts/运行V12门控独立性诊断.sh
bash -n scripts/运行V14固定相关平滑.sh
bash -n scripts/运行G5三维框组件误差归因.sh
bash -n scripts/运行V20立体几何深度残差.sh
bash -n scripts/运行V22轴向IoU深度损失.sh
bash -n scripts/运行V09_trainval官网最终提交.sh

python tools/检查QECR消融配置.py
python tools/检查V09Trainval官网复核配置.py
python tools/测试固定轮次Checkpoint.py
python tools/诊断3D质量排序_G1.py --self_test
python tools/诊断三维框组件误差_G5.py --self_test
python tools/诊断朝向感知几何深度_G6.py --self_test
python tools/诊断深度损失三维IoU敏感性_G7.py --self_test
python tools/测试V20立体几何深度残差.py
python tools/汇总V20立体几何深度残差.py --self_test
python -m py_compile tools/审计V20训练参数.py
python tools/测试V22轴向IoU深度损失.py
python tools/汇总V22轴向IoU深度损失.py --self_test
python -m py_compile tools/检查V22模型结构.py tools/审计V22训练参数.py
python tools/测试多候选视差监督.py
python tools/测试3D质量排序.py
python tools/测试Checkpoint兼容白名单.py
python tools/汇总V09强基线质量组合.py --self_test
python tools/测试前景TopK深度读出.py
python tools/汇总V10前景TopK深度诊断.py --self_test
python tools/测试V11动态深度上采样.py
python tools/汇总V11动态深度上采样.py --self_test
python tools/诊断分组相关信息_V12G0.py --self_test
python tools/测试V12轻量分组相关.py
python tools/汇总V12分组相关门控.py --self_test
python tools/汇总V12时延.py --self_test
python tools/汇总V12双种子复核.py --self_test
python tools/汇总V12门控移植诊断.py --self_test
python tools/测试V14固定相关平滑.py
python tools/汇总V14固定相关平滑.py --self_test
python tools/测试空间投影对齐.py
python tools/测试查询深度模块.py
python tools/测试QECR核心模块_最终.py
python tools/测试P2P3双目标定.py
python tools/测试QECR包装集成.py
python tools/测试PyTorch2完整模型导入.py
python -m py_compile \
  lib/models/monodetr/query_epipolar_runtime.py \
  lib/models/monodetr/query_depth.py \
  lib/models/monodetr/quality_ranking.py \
  lib/models/monodetr/geometry_alignment.py \
  lib/models/monodetr/geometry_depth_residual.py \
  lib/models/monodetr/axial_depth_iou.py \
  lib/models/monodetr/depth_predictor/disparityloss.py \
  lib/models/monodetr/depth_predictor/depth_readout.py \
  lib/models/monodetr/depth_predictor/depth_predictor_lightstereo.py \
  lib/models/monodetr/stereodetr.py \
  lib/models/qecr_model.py \
  lib/datasets/kitti/qecr_kitti_dataset.py \
  lib/helpers/qecr_dataloader_helper.py \
  lib/helpers/qecr_model_helper.py \
  lib/helpers/qecr_trainer_helper.py \
  lib/helpers/optimizer_helper.py \
  lib/helpers/save_helper.py \
  lib/helpers/trainer_helper.py \
  tools/诊断3D质量排序_G1.py \
  tools/诊断三维框组件误差_G5.py \
  tools/诊断朝向感知几何深度_G6.py \
  tools/诊断深度损失三维IoU敏感性_G7.py \
  tools/测试V20立体几何深度残差.py \
  tools/检查V20模型结构.py \
  tools/汇总V20立体几何深度残差.py \
  tools/测试V22轴向IoU深度损失.py \
  tools/检查V22模型结构.py \
  tools/审计V22训练参数.py \
  tools/汇总V22轴向IoU深度损失.py \
  tools/测试3D质量排序.py \
  tools/测试Checkpoint兼容白名单.py \
  tools/汇总V09强基线质量组合.py \
  tools/测试前景TopK深度读出.py \
  tools/汇总V10前景TopK深度诊断.py \
  tools/测试V11动态深度上采样.py \
  tools/检查V11模型结构.py \
  tools/汇总V11动态深度上采样.py \
  tools/诊断分组相关信息_V12G0.py \
  tools/测试V12轻量分组相关.py \
  tools/检查V12模型结构.py \
  tools/审计V12训练参数.py \
  tools/汇总V12分组相关门控.py \
  tools/测试V14固定相关平滑.py \
  tools/汇总V14固定相关平滑.py \
  tools/测试空间投影对齐.py \
  tools/训练与评估_无蒸馏QECR.py
python -m py_compile \
  tools/检查V09Trainval官网复核配置.py \
  tools/测试固定轮次Checkpoint.py \
  tools/检查并打包KITTI提交.py

echo "QECR pre-training checks passed"
