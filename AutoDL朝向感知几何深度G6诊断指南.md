# AutoDL朝向感知几何深度G6诊断指南

## 诊断目标

V20已经证明：在冻结V09的条件下，仅增加查询级深度残差头不能提升正式KITTI验证集精度。下一步不应继续扫描残差头超参数，而应先检查它接收的几何深度先验是否存在系统误差。

StereoDETR当前代码中的几何深度为：

```text
z_geo = f × h_3d / h_2d + l / 2
```

其中 `l/2` 对所有朝向都相同。但KITTI三维框绕相机Y轴旋转后，沿相机Z轴的半投影实际为：

```text
(|l × sin(ry)| + |w × cos(ry)|) / 2
```

G6只诊断这处差异是否能解释V20失败，不训练模型，也不修改checkpoint。

## 为什么必须拆成校准集与留出集

脚本把3769个验证样本按编号奇偶固定拆分：

- 偶数编号：校准集，只用于给每个类别拟合一个裁剪到 `[-1,1]` 的标量；
- 奇数编号：留出集，只用于误差统计与KITTI 3D AP_R40比较；
- 自动结论只看留出集。

这仍然不是正式模型实验，因为校准阶段使用了部分验证集真值。它只决定是否值得实现V21。正式V21必须在训练集学习，再在完整验证集上评估。

## 上传后执行

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export OMP_NUM_THREADS=8

python tools/诊断朝向感知几何深度_G6.py --self_test

mkdir -p 'outputs/朝向感知几何深度诊断_G6'
set -o pipefail

python tools/诊断朝向感知几何深度_G6.py \
  --output_dir 'outputs/朝向感知几何深度诊断_G6' \
  2>&1 | tee 'outputs/朝向感知几何深度诊断_G6/运行.log'

G6_EXIT=${PIPESTATUS[0]}
echo "G6退出码：$G6_EXIT"
cat 'outputs/朝向感知几何深度诊断_G6/诊断报告.md'
```

该诊断只读已有文本预测，主要运行KITTI评估，通常不需要GPU，也不会产生checkpoint。

## 回传哪些结果

请回传下面命令的完整输出：

```bash
cat 'outputs/朝向感知几何深度诊断_G6/诊断报告.md'

grep -E \
  '自动结论:|解释:|Traceback|CUDA out of memory|G6诊断完成' \
  'outputs/朝向感知几何深度诊断_G6/运行.log' \
  || true
```

## 预注册止损线

只有同时满足以下条件，才允许开发V21短程配对版本：

1. 校准和留出匹配数均不少于3000；
2. 最佳可实现候选在留出集的Car 3D AP_R40 Moderate至少提升0.20；
3. Car深度MAE至少改善1%；
4. Car Easy/Hard均不得下降0.10以上；
5. Pedestrian/Cyclist的Moderate与Hard均不得下降0.25以上。

若未通过，就停止几何深度公式与残差头路线，不再为V20调学习率、轮数或头宽度。
