# AutoDL轻量三维质量排序V06实验指南

## 1. 这次为什么做V06

G1已经排除“只调sigma指数”作为主要方案：最佳`gamma=0.5`仅比V00提高0.0981 AP。但同一批Top-K检测若用真实3D IoU排序，Car/Pedestrian/Cyclist Moderate分别仍有8.4768、10.4370、6.2102 AP的诊断上限。因此V06针对的是**已有候选框的3D质量没有被最终置信度正确表达**，不是继续加大骨干或深度网络。

V06不换ResNet50。检测器从V00最优权重加载后全部冻结，只训练66,561参数的查询级质量头。训练期使用3D IoU；推理期不计算IoU。

## 2. 版本作用

| 版本 | 点式3D IoU软目标 | 成对排序 | 用途 |
|---|---:|---:|---|
| V00 | 否 | 否 | 固定分母48.3226 |
| V06B | 是 | 否 | 后续论文消融 |
| V06C | 是 | 是 | 优先运行的完整候选版本 |

第一次先跑V06C，不先跑V06B。若完整机制都不能改善，点式弱版本通常没有优先花费租卡费用的理由；若V06C通过，再补V06B拆分成对排序的净作用。

## 3. 上传后先检查

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="8.9"

bash scripts/运行QECR测试.sh
echo "测试退出码：$?"
```

必须看到：

```text
15 个无蒸馏正式消融配置检查通过
V06 质量头、点式/成对损失、3D目标和解码融合测试通过
QECR pre-training checks passed
测试退出码：0
```

若这里失败，不开始计费训练。

## 4. 一轮冒烟

正式配置是5轮。项目已经准备好继承正式开关的一轮冒烟配置，不要复制后破坏相对继承路径：

```bash
mkdir -p 'outputs/三维质量排序冒烟/V06C_一轮'
set -o pipefail
python tools/训练与评估_无蒸馏QECR.py \
  --config 'configs/AutoDL_V06C_一轮冒烟.yaml' \
  2>&1 | tee 'outputs/三维质量排序冒烟/V06C_一轮/控制台.log'

echo "冒烟退出码：${PIPESTATUS[0]}"
```

日志应明确显示V00预训练权重只缺少`quality_head.*`新参数，`loss_quality_point`和`loss_quality_pair`均为有限值，训练、验证、保存和回载全部成功。

第一次训练批次会触发KITTI旋转3D IoU的Numba/CUDA编译，可能明显慢于后续批次；只要GPU进程仍在且没有异常栈，不要在第一批立刻中断。

## 5. 正式运行V06C五轮

```bash
screen -S stereodetr_v06c
```

进入screen后：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="8.9"

mkdir -p 'outputs/三维质量排序短程筛选/V06C_点式加成对排序'
set -o pipefail

python tools/训练与评估_无蒸馏QECR.py \
  --config 'versions/V06C_点式加成对排序/config.yaml' \
  2>&1 | tee 'outputs/三维质量排序短程筛选/V06C_点式加成对排序/控制台.log'

echo "V06C退出码：${PIPESTATUS[0]}"
```

按`Ctrl+A`再按`D`退出screen。查看状态：

```bash
screen -ls
tail -n 80 'outputs/三维质量排序短程筛选/V06C_点式加成对排序/控制台.log'
```

## 6. 结果门槛

比较同一V00的`48.3226`，不是另一个工程的48.9152：

- `V06C ≥ 48.6226`且Hard下降不超过0.30：通过，补跑V06B；
- `48.4226 ≤ V06C < 48.6226`：第二随机种子复核；
- `V06C < 48.4226`或多难度明显退化：停止，不跑195轮。

种子444的五轮实测最佳值为`48.4924`（第5轮），相对V00提高`+0.1698 AP`，因此进入预定的第二随机种子复核，不直接补V06B。上传`configs/AutoDL_V06C_种子445复核.yaml`后运行：

```bash
mkdir -p 'outputs/三维质量排序短程筛选/V06C_种子445复核'
set -o pipefail

python tools/训练与评估_无蒸馏QECR.py \
  --config 'configs/AutoDL_V06C_种子445复核.yaml' \
  2>&1 | tee \
  'outputs/三维质量排序短程筛选/V06C_种子445复核/控制台.log'

V06C_SEED445_EXIT=${PIPESTATUS[0]}
echo "V06C种子445退出码：$V06C_SEED445_EXIT"
```

复核决策在运行前固定：种子445也提高至少`+0.10 AP`，且两个种子的平均提升至少`+0.10 AP`，才补V06B；种子445不升且两种子平均不足`+0.10 AP`，则停止该实现。第二种子完成前不临时延长训练轮数。

通过后，V06B的命令只需把配置替换为：

```bash
python tools/训练与评估_无蒸馏QECR.py \
  --config 'versions/V06B_点式三维质量排序/config.yaml'
```

## 7. 实时性口径

质量头理论上只对50个推理query增加约3.29M MAC，但不能据此宣称“零时延”。V06C通过精度门槛后，必须用项目已有的同步时延脚本分别测V00和V06C，同一张4090、同一batch=1、相同warm-up和重复次数，再报告平均时延与FPS。

## 8. 依据与边界

模块依据来自RARE的公开论文和官方代码：独立质量头、3D IoU点式目标以及同类成对排序。这里是针对StereoDETR G1实测痛点的迁移实现，不宣称RARE在单目任务中的提升会原样出现在StereoDETR上。

- 论文：<https://openaccess.thecvf.com/content/CVPR2026/html/Park_RARE_Learn_to_RAnk_and_REtrieve_for_Monocular_3D_Object_CVPR_2026_paper.html>
- 官方代码：<https://github.com/HyeonjeongPark37/RARE>
