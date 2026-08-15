# AutoDL深度损失三维IoU敏感性G7诊断指南

## 为什么做G7

G5证明最终轴向深度仍是V09最大的三维框组件瓶颈，但V20和G6共同证明：二维框高、预测尺寸和朝向构造的几何深度不能可靠预测真实深度残差。因此不再设计新的末端残差头。

当前StereoDETR的查询深度损失本质为带预测方差的米制绝对误差：

```text
sqrt(2) × exp(-log_variance) × |pred_depth - gt_depth| + log_variance
```

同样的0.3米误差，对不同长度、朝向、类别和距离目标造成的3D IoU影响并不相同。G7先检查一个更靠近正式指标、且只增加训练成本的候选是否有立项证据。

## G7做什么

G7不训练、不加载模型、不改checkpoint。它读取V09的3769个验证预测，进行同类别2D一对一匹配，然后构造两类GT Oracle：

- 比例修正：只消除真实深度残差的10%、25%、50%；
- 限幅修正：沿真实方向最多修正0.10、0.25、0.50米。

同时计算：

- 完整验证集KITTI 3D AP_R40；
- 原始米制误差、轴向尺寸归一化误差、轴向区间IoU损失与完整3D IoU损失的Spearman相关性；
- Car/Pedestrian/Cyclist跨越正式0.70/0.50 3D IoU阈值的对象数；
- Car近、中、远距离分层。

所有Oracle都用了验证集GT，只能决定是否实现V22训练损失，不能写成模型精度。

## 上传后执行

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

python tools/诊断深度损失三维IoU敏感性_G7.py --self_test
```

必须先看到：

```text
G7自检通过：轴向半投影、区间IoU、比例修正与限幅修正正常
```

再运行正式诊断：

```bash
mkdir -p 'outputs/深度损失三维IoU敏感性诊断_G7'
set -o pipefail

python tools/诊断深度损失三维IoU敏感性_G7.py \
  --output_dir 'outputs/深度损失三维IoU敏感性诊断_G7' \
  2>&1 | tee 'outputs/深度损失三维IoU敏感性诊断_G7/运行.log'

G7_EXIT=${PIPESTATUS[0]}
echo "G7退出码：$G7_EXIT"
cat 'outputs/深度损失三维IoU敏感性诊断_G7/诊断报告.md'
```

该脚本会调用项目正式KITTI 3D IoU CUDA算子和官方评估，所以保留GPU环境；它不做模型前向，预计远短于一次训练。

## 预注册门槛

主判定只允许比较两个最弱Oracle：`GT残差10%`和`GT方向最大0.10m`，不能看到结果后改用25%、50%宣称通过。

V22只有同时满足以下条件才允许实现：

1. 主候选Car Moderate至少提升0.20 AP；
2. Car Easy/Hard以及Pedestrian/Cyclist Moderate/Hard均不得下降0.10 AP以上；
3. Car轴向IoU损失与完整3D IoU损失的Spearman比原始米制误差至少提高0.03；
4. 主候选在Car匹配对象上净跨0.70 IoU阈值至少20个。

通过也只允许实现训练期辅助损失，并做V22O/V22A五轮同起点、同种子公平配对。该候选不修改推理结构，理论推理参数和时延增量为0，但仍需实验核对。

## 回传结果

```bash
cat 'outputs/深度损失三维IoU敏感性诊断_G7/诊断报告.md'

grep -E \
  'G7诊断完成|自动结论:|解释:|Traceback|CUDA out of memory|nan' \
  'outputs/深度损失三维IoU敏感性诊断_G7/运行.log' \
  || true
```
