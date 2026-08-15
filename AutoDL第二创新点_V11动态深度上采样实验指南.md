# AutoDL 第二创新点候选 V11 动态深度上采样实验指南

> 最终状态（2026-08-11）：**V11A已被正式否决，请勿运行V11B。** 五周期同起点配对中，V11A相对V11O的Car 3D AP_R40 Moderate为`48.5817`对`49.6485`，下降`1.0668 AP`。本指南保留用于复现实验与审计，不再作为待执行计划。

更新时间：2026-08-11

## 1. 本轮只验证什么

V11只修改 StereoDETR 深度分类分支的固定双线性上采样：

- V11O：原双线性上采样，续训原深度分类分支；
- V11A：只把第一级上采样换成零偏移初始化的动态采样；
- V11B：两级上采样都替换，但只有V11A通过门槛后才允许运行。

三组都从同一个V09最佳checkpoint开始。V09质量头保留用于推理排序，但被冻结且不计算质量损失；除`depth_predictor.depth_classifier.*`外的参数全部冻结。原深度分类卷积学习率为`2e-5`，新增偏移层为`2e-4`。

## 2. 上传时必须保护服务器产物

将本地 `QECR-StereoDETR` 更新同步到：

```text
/root/autodl-tmp/QECR-StereoDETR
```

不要删除或覆盖服务器上的 `outputs/`，尤其是：

```text
outputs/最终组合_V09/V09_V08O加点式三维质量排序/
```

上传后检查新文件：

```bash
cd /root/autodl-tmp/QECR-StereoDETR

ls -lh \
  'versions/V11O_双线性深度上采样控制/config.yaml' \
  'versions/V11A_第一级动态深度上采样/config.yaml' \
  'versions/V11B_两级动态深度上采样/config.yaml' \
  'scripts/运行V11动态深度上采样.sh' \
  'tools/测试V11动态深度上采样.py'
```

## 3. 环境与V09权重门禁

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

V09_CKPT='outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth'

ls -lh "$V09_CKPT"
sha256sum "$V09_CKPT"
```

已记录的正式V09 SHA256应为：

```text
6b673fef2a54e2a5fbc44be731b24914770fffebaf1c20a0cacc43228dd316b4
```

不一致时停止，不要开始训练。

## 4. 完整训练前测试

```bash
set -o pipefail

bash scripts/运行QECR测试.sh \
  2>&1 | tee V11训练前测试.log

echo "测试退出码：${PIPESTATUS[0]}"
```

必须同时看到：

```text
3 个V11动态深度上采样配置检查通过
V11动态深度上采样测试通过
V11结果解析与预注册止损判定自检通过
QECR pre-training checks passed
测试退出码：0
```

## 5. 24样本一轮冒烟

脚本会在缺失时自动从正式`train.txt`取前24个编号生成`train_v11_smoke.txt`，不会改动正式划分。

```bash
bash scripts/运行V11动态深度上采样.sh smoke
```

检查：

```bash
grep -E \
  'V11模型结构与训练范围检查通过|loss_depth_map:|Best Result:|Traceback|CUDA out of memory|nan|inf' \
  'outputs/第二创新点_V11动态深度上采样/冒烟测试/控制台.log'
```

冒烟AP只来自24个样本，完全不能作为精度结论。这里只确认：checkpoint白名单通过、动态偏移进入图、深度损失有限、前反向与评估链完整。

## 6. 正式V11O/V11A五周期配对

确认GPU空闲且输出目录不存在后进入screen：

```bash
screen -S v11_pair
```

在screen中执行：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

bash scripts/运行V11动态深度上采样.sh pair
```

按`Ctrl+A`、再按`D`退出screen。查看进度：

```bash
screen -ls

tail -n 80 \
  'outputs/第二创新点_V11动态深度上采样/V11O_双线性训练控制/控制台.log'

tail -n 80 \
  'outputs/第二创新点_V11动态深度上采样/V11A_第一级动态上采样/控制台.log'
```

脚本按V11O→V11A顺序运行，不会并行争抢4090显存，并自动生成：

```text
outputs/第二创新点_V11动态深度上采样/V11配对结果.md
```

## 7. 是否运行V11B

读取报告：

```bash
cat 'outputs/第二创新点_V11动态深度上采样/V11配对结果.md'
```

只有报告出现：

```text
V11A自动判定：精度通过
```

并且后续正式时延没有明显异常，才运行：

```bash
screen -S v11_stage2
```

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open
export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

bash scripts/运行V11动态深度上采样.sh stage2
```

脚本会再次检查V11A报告；没有达到“精度通过”时会拒绝启动V11B。

## 8. 正式门槛

V11A相对V11O：

- Car 3D AP_R40 Moderate至少`+0.15 AP`；
- Car Hard不低于`-0.10 AP`；
- Pedestrian与Cyclist Moderate不能同时下降；
- Median同步模型前向时延增加不超过`3%`。

Car Moderate只有`+0.05—+0.15 AP`时判为“谨慎”，先补复评或独立种子，不运行V11B。低于`+0.05 AP`或出现明显跨类别退化则停止该路线。

## 9. 跑完后回传

```bash
cd /root/autodl-tmp/QECR-StereoDETR

cat 'outputs/第二创新点_V11动态深度上采样/V11配对结果.md'

for info in \
  'outputs/第二创新点_V11动态深度上采样/V11O_双线性训练控制/运行信息.txt' \
  'outputs/第二创新点_V11动态深度上采样/V11A_第一级动态上采样/运行信息.txt'
do
  echo "===== $info ====="
  cat "$info"
done

for structure in \
  'outputs/第二创新点_V11动态深度上采样/V11O_双线性训练控制/模型结构检查.txt' \
  'outputs/第二创新点_V11动态深度上采样/V11A_第一级动态上采样/模型结构检查.txt'
do
  echo "===== $structure ====="
  cat "$structure"
done
```

将这些输出发回后，再决定是否测正式时延、运行V11B或停止路线。
