# AutoDL 官方 Batch 16 复核训练与 KITTI 提交指南

## 1. 本轮目的

本轮只修正训练协议，不增加新的检测模块，也不改动既有 QLQC 结构。旧的 V09 配置、权重和输出目录均保留。

新协议固定为：

1. 使用 KITTI `trainval` 的 7481 组有标注双目图像，以真实全局批量 16 训练 StereoDETR 195 轮。
2. 从固定的第 195 轮权重出发，执行 1 轮低学习率强控制训练。
3. 冻结检测器，只训练既有 QLQC 质量分支 3 轮。
4. 分别保留 StereoDETR 基础模型和 QLQC 最终模型的 KITTI test 推理与打包能力。

这里的 `batch_size=16` 是一次优化更新实际使用的全局样本数，不是梯度累积形成的名义批量。双卡档位会切分为每卡 8 个样本，同时确保图像、标定、尺寸和逐样本标签使用同一切分边界。

## 2. 三种运行档位

| 档位 | GPU | 精度 | 全局批量 | 用途 |
|---|---:|---|---:|---|
| `single_fp32` | 1 | FP32 | 16 | 第一选择，最接近官方单卡训练数值协议 |
| `single_amp` | 1 | FP16 AMP与动态缩放 | 16 | 单卡 FP32 显存不足时使用 |
| `dual_fp32` | 2 | FP32 | 16，即8+8 | 单卡方案失败后租两张卡使用 |

不要把三个档位混在同一训练目录中续跑。脚本会写入训练档位锁并主动阻止混用。

## 3. 上传后先做预检

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export OMP_NUM_THREADS=8

bash scripts/运行官方批量16_QLQC最终提交.sh preflight single_fp32
```

预检必须出现以下三行：

```text
官方批量16协议检查通过
批次切分检查通过
官方批量16训练预检通过
```

## 4. 用两次更新检查显存和数值稳定性

先运行单卡 FP32：

```bash
bash scripts/运行官方批量16_QLQC最终提交.sh smoke single_fp32
```

出现 `批量16冒烟通过` 后才开始长训练。

如果明确出现 `CUDA out of memory`，改用单卡 AMP：

```bash
bash scripts/运行官方批量16_QLQC最终提交.sh smoke single_amp
```

如果 AMP 因算子兼容性失败，或者希望坚持 FP32，再租两张 GPU，并运行：

```bash
bash scripts/运行官方批量16_QLQC最终提交.sh smoke dual_fp32
```

双卡预检输出应显示两张可见 GPU，训练日志应包含：

```text
Global batch size: 16
Nominal per-device chunks: [8, 8]
AMP training: False
```

## 5. 后台运行完整训练

下面以通过冒烟的 `single_fp32` 为例。若实际通过的是其他档位，只替换最后一个档位参数。

```bash
screen -dmS qlqc_batch16 bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行官方批量16_QLQC最终提交.sh all single_fp32
'
```

双卡时将最后一行改为：

```bash
bash scripts/运行官方批量16_QLQC最终提交.sh all dual_fp32
```

脚本按顺序完成基础模型训练、基础模型测试集推理、强控制训练、QLQC 训练、QLQC 测试集推理和 ZIP 打包。训练中断后，用完全相同的档位重新执行同一命令即可从每轮保存的 `checkpoint.pth` 继续。

## 6. 查看进度和预计剩余时间

```bash
screen -ls

bash scripts/运行官方批量16_QLQC最终提交.sh status single_fp32

tr '\r' '\n' \
  < 'outputs/KITTI_trainval官方批量16复核/T0_StereoDETR/控制台.log' \
  | tail -n 40
```

训练进度中的 `epochs` 行会显示已完成轮数、已运行时间和预计剩余时间。显卡状态可用下面的命令查看：

```bash
nvidia-smi
```

## 7. 分阶段执行方式

如果不想一次跑完，可依次执行：

```bash
bash scripts/运行官方批量16_QLQC最终提交.sh train_base single_fp32
bash scripts/运行官方批量16_QLQC最终提交.sh infer_base single_fp32
bash scripts/运行官方批量16_QLQC最终提交.sh train_qlqc single_fp32
bash scripts/运行官方批量16_QLQC最终提交.sh infer_qlqc single_fp32
```

KITTI 每个榜单存在提交频率限制。两个 ZIP 都会保留用于审计，但优先提交最终 QLQC 包，不要在提交次数不足时先消耗基础模型名额。

## 8. 输出位置

基础模型固定权重：

```text
outputs/KITTI_trainval官方批量16复核/T0_StereoDETR/stereodetr_trainval_global_batch16/checkpoint_final.pth
```

QLQC 最终固定权重：

```text
outputs/KITTI_trainval官方批量16复核/T2_QLQC/qlqc_trainval_global_batch16/checkpoint_final.pth
```

基础模型提交包：

```text
outputs/KITTI_trainval官方批量16复核/StereoDETR_global_batch16_KITTI_test_submission.zip
```

最终 QLQC 提交包：

```text
outputs/KITTI_trainval官方批量16复核/QLQC_global_batch16_KITTI_test_submission.zip
```

最终提交前执行：

```bash
ZIP_FILE='outputs/KITTI_trainval官方批量16复核/QLQC_global_batch16_KITTI_test_submission.zip'
unzip -t "$ZIP_FILE" | tail -n 3
unzip -Z1 "$ZIP_FILE" | wc -l
cat "$ZIP_FILE.sha256"
```

应得到 ZIP 无错误、文件数 7518，且文件直接位于 ZIP 根目录。

## 9. 结果判定

当前需要跨越的官方 StereoDETR Car 3D AP 差距为 Easy `0.36`、Moderate `0.42`、Hard `0.21`。本轮首先观察最终 QLQC 的 KITTI test Car Moderate 是否超过 `41.17`，同时保留三类完整的 Easy、Moderate、Hard 结果。

本轮没有修改 QLQC 方法本身。因此，如果结果提升，应归因于训练批量和官方训练协议复现质量的改善，再报告 QLQC 相对同协议 StereoDETR 的增益。不要把训练协议变化包装成新的网络创新点。
