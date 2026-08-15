# AutoDL 第二创新点候选：质量引导候选检索 G3 诊断指南

## 1. 为什么先做无训练诊断

V06B 已经证明查询级三维质量可以改善最终排序，但当前代码先按分类概率选择 Top-K，再把深度置信度和学习质量乘到入选结果上。因此，一个分类分数略低但三维定位明显更好的查询如果没有进入 Top-K，V06B 的质量头无法把它召回。

V07 将候选选择分数从

```text
分类概率
```

改为与最终排序一致的

```text
分类概率 × exp(-深度sigma) × 三维质量概率^1.5
```

预测框、质量头权重、Top-K 数量和最终融合公式均不变。该诊断不训练、不新增参数，一次评估约为此前完整验证的耗时。只有 G3 通过预设门槛，才把它确认为第二创新点。

## 2. 上传后先运行测试

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

bash scripts/运行QECR测试.sh
echo "测试退出码：$?"
```

预期看到：

```text
16 个无蒸馏正式消融配置检查通过
V06质量头/监督与V07质量引导Top-K检索测试通过
QECR pre-training checks passed
测试退出码：0
```

## 3. 固定同一个 V06B 权重

```bash
cd /root/autodl-tmp/QECR-StereoDETR

V06B_CKPT='/root/autodl-tmp/QECR-StereoDETR/outputs/三维质量排序短程筛选/V06B_点式三维质量排序/qecr_v06b_pointwise_3d_quality/checkpoint_best.pth'

ls -lh "$V06B_CKPT"
sha256sum "$V06B_CKPT"
```

如果文件不存在，不要训练 V07，先找到实际的 V06B 最优权重路径并修改变量。

## 4. 重新跑同链路对照 G3-A

该组保持原始“分类概率 Top-K”，但固定最终质量指数为 1.5，用于确认修改后代码仍精确复现 G2 的 `48.5477`。

```bash
mkdir -p 'outputs/质量引导候选检索诊断_G3'
set -o pipefail

python tools/训练与评估_无蒸馏QECR.py \
  --evaluate_only \
  --quality_score_power 1.5 \
  --config 'versions/V06B_点式三维质量排序/config.yaml' \
  2>&1 | tee 'outputs/质量引导候选检索诊断_G3/G3A_分类分数TopK.log'

echo "G3-A退出码：${PIPESTATUS[0]}"
```

G3-A 的 Car 3D AP_R40@0.70 应为：

```text
69.0256, 48.5477, 41.6568
```

若不能复现，停止 G3，先检查权重、数据划分和代码版本。

## 5. 为 V07 建立独立评估目录

V07 不训练新权重，只复用同一个 V06B checkpoint。复制权重是为了让输出目录、日志和检测结果与 V06B 隔离。

```bash
V07_DIR='/root/autodl-tmp/QECR-StereoDETR/outputs/质量引导候选检索诊断_G3/V07_质量引导候选检索/qecr_v07_quality_guided_topk'

mkdir -p "$V07_DIR"
cp -p "$V06B_CKPT" "$V07_DIR/checkpoint_best.pth"

sha256sum \
  "$V06B_CKPT" \
  "$V07_DIR/checkpoint_best.pth"
```

两个 SHA256 必须完全相同。

## 6. 运行质量引导 Top-K：G3-B

```bash
set -o pipefail

python tools/训练与评估_无蒸馏QECR.py \
  --evaluate_only \
  --config 'versions/V07_质量引导候选检索/config.yaml' \
  2>&1 | tee 'outputs/质量引导候选检索诊断_G3/G3B_质量引导TopK.log'

G3B_EXIT=${PIPESTATUS[0]}
echo "G3-B退出码：$G3B_EXIT"
```

## 7. 自动提取对照结果

```bash
for log_file in \
  'outputs/质量引导候选检索诊断_G3/G3A_分类分数TopK.log' \
  'outputs/质量引导候选检索诊断_G3/G3B_质量引导TopK.log'
do
  echo "===== $(basename "$log_file") ====="
  awk '
    /Car AP_R40@0.70, 0.70, 0.70:/ {inside=1; next}
    inside && /3d   AP:/ {print; exit}
  ' "$log_file"
done
```

同时保留 Pedestrian 和 Cyclist 的完整输出，不得只看 Car Moderate。

## 8. 预注册判断门槛

以 G3-A 的 Car 3D AP_R40 Moderate 为分母，并检查 Hard：

- G3-B 的 Moderate 提升至少 `+0.10 AP`，且 Hard 下降不超过 `0.10 AP`：通过，可将“质量引导候选检索”发展为第二创新点，并补充多类别、时延和候选变化率分析。
- Moderate 提升 `+0.03～+0.10 AP`：谨慎，只能算对第一创新点的推理细化，暂不单列第二创新点。
- Moderate 提升小于 `+0.03 AP`、为负，或 Hard 明显下降：否决，不投入训练费用，转向新的痛点。

该门槛在看到 G3-B 结果前固定。不能因为某一个类别或 Easy 指标偶然提高而临时改变主指标。

## 9. 这个版本与 RARE 的关系

V07 受 RARE“按定位质量检索预测”的思想启发，但不是 RARE 完整的多假设 Retrieve 模块。RARE 为每个对象构造多个三维假设再检索；V07 只在 StereoDETR 已有查询集合中，使 Top-K 候选选择与最终三维质量评分一致。论文中必须按这个边界表述，不能宣称复现或完整移植了 RARE。

## 10. 2026-08-09 实际结果与最终决定

| 类别与难度（3D AP_R40） | G3-A | G3-B | 变化 |
|---|---:|---:|---:|
| Car Easy / Moderate / Hard | 69.0256 / 48.5477 / 41.6568 | 69.0256 / 48.5481 / 41.6572 | +0.0000 / +0.0004 / +0.0004 |
| Pedestrian Moderate / Hard | 26.5237 / 21.3299 | 26.5351 / 21.3414 | +0.0114 / +0.0115 |
| Cyclist Moderate / Hard | 23.5761 / 22.0161 | 23.5761 / 22.0161 | +0.0000 / +0.0000 |

结论：**否决V07**。Car Moderate仅提升`+0.0004 AP`，未达到预注册的`+0.10 AP`门槛。V07不训练、不进入正式消融表，也不通过继续扫描Top-K或阈值追逐验证集小数点波动。该版本只作为一次有效的负向诊断保留。
