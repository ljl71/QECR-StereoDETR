# AutoDL 三维质量排序 G1 诊断指南

## 这一步解决什么问题

G1 不训练模型，也不修改 V00 权重。它用已经复现完成的 V00 最优权重跑一次 KITTI 验证集，回答两个问题：

1. 当前的分类分数和深度置信度能否把 3D IoU 更高的框排到前面；
2. 如果使用理想的 3D IoU 质量分数，Car 3D AP_R40 Moderate 还有多少可提升空间。

只有 G1 通过，才实现和训练 V06B/V06C。这样可以避免再次直接支付 195 轮训练费用。

## 必须使用的权重

使用 V00 官方基线自己的最优权重：

```text
/root/autodl-tmp/QECR-StereoDETR/outputs/QECR消融实验/V00_官方基线/qecr_v00_official_baseline/checkpoint_best.pth
```

不要使用 V01、V02、V03 或 V05 的权重。

## 第一步：上传本次修改

把本地项目重新同步到服务器：

```text
/root/autodl-tmp/QECR-StereoDETR
```

至少确认下面两个新文件已经出现：

```bash
cd /root/autodl-tmp/QECR-StereoDETR

ls -lh \
  tools/诊断3D质量排序_G1.py \
  AutoDL三维质量排序G1诊断指南.md
```

同步时不要删除服务器上的 `outputs`，也不要覆盖 V00 的 `checkpoint_best.pth`。

## 第二步：恢复环境

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
```

## 第三步：只运行自检

```bash
python tools/诊断3D质量排序_G1.py --self_test
```

预期输出：

```text
G1 自检通过：指数评分、止损门槛和相关性计算正常
```

## 第四步：确认 V00 权重

```bash
V00_CKPT='/root/autodl-tmp/QECR-StereoDETR/outputs/QECR消融实验/V00_官方基线/qecr_v00_official_baseline/checkpoint_best.pth'

ls -lh "$V00_CKPT"
```

如果这里显示文件不存在，先用下面的命令寻找，不要启动诊断：

```bash
find /root/autodl-tmp/QECR-StereoDETR/outputs \
  -type f -name checkpoint_best.pth -print
```

## 第五步：运行完整 G1

完整验证集只做一次模型推理，通常是分钟级，不是训练任务，不需要 screen。

```bash
mkdir -p outputs/三维质量排序诊断_G1
set -o pipefail

python tools/诊断3D质量排序_G1.py \
  --config 'versions/V00_官方基线/config.yaml' \
  --checkpoint "$V00_CKPT" \
  --output_dir 'outputs/三维质量排序诊断_G1' \
  2>&1 | tee outputs/三维质量排序诊断_G1/运行.log

G1_EXIT=${PIPESTATUS[0]}
echo "G1退出码：$G1_EXIT"
```

退出码必须是 `0`。

## 第六步：查看报告

```bash
cat outputs/三维质量排序诊断_G1/诊断报告.md
```

主要看三个值：

- `gamma=1`：当前 StereoDETR 的 `分类分数 × 深度置信度`；
- 最佳 gamma：只改变 sigma 权重、无需训练时的最好结果；
- 3D IoU Oracle：使用验证集真值进行理想排序的诊断上限。

## 自动止损规则

### 结论为“否决”

Oracle 相对 gamma=1 提升小于 0.3 AP。停止质量排序路线，不训练 V06。

### 结论为“先做零训练校准”

某个 gamma 相对 gamma=1 已提升至少 0.3 AP。先复核该 gamma 和完整三类结果，再决定是否需要新增训练损失。

### 结论为“通过”

现有 sigma 指数无法明显提升，但 Oracle 至少提升 0.8 AP。说明现有候选里存在好框，只是当前分数不会正确排序。下一步实现：

1. V06A：相同权重、相同轮数的续训对照；
2. V06B：真实 3D IoU 软质量目标，推理结构不变；
3. V06C：V06B 加 RARE 风格 pair-wise 排序损失，推理结构仍不变。

### 结论为“谨慎”

Oracle 提升位于 0.3–0.8 AP，只允许进行同起点的短程配对实验，不能直接跑 195 轮。

## 结果文件

```text
outputs/三维质量排序诊断_G1/
├── 运行.log
├── 诊断报告.md
├── 诊断汇总.json
├── 检测质量明细.csv
├── 指数扫描/
└── Oracle_仅诊断禁止提交/
```

`Oracle_仅诊断禁止提交` 使用了验证集真值，严禁把它当作论文结果、测试集结果或模型输出。

