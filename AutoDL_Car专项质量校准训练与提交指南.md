# AutoDL Car 专项质量校准训练与 KITTI 提交指南

## 1. 实验目的

本实验以已经完成的全量 `trainval`、全局批量 16 的 QLQC 权重为固定起点，不重新训练 195 轮基础检测器。旧模型全部冻结，只训练一个零初始化的 Car 类别残差质量头 3 轮，最后在 KITTI `test` 的 7518 组图像上生成一次候选提交包。

新分支只调整最终标签为 Car 的排序质量分数，不改变候选框、类别分数、深度、尺寸和朝向，也不改写 Pedestrian 与 Cyclist 的原 QLQC 分数。新增参数为 8257 个。

这是没有公开验证标签条件下的一次固定协议实验。训练轮数、损失、残差幅度与推理规则均已写入配置，不应根据官网测试结果反复改动。

## 2. 上传前检查

从 PyCharm 上传整个最新项目到：

```text
/root/autodl-tmp/QECR-StereoDETR
```

不要删除已有输出，尤其需要保留下面的全量 QLQC 起点：

```text
outputs/KITTI_trainval官方批量16复核/T2_QLQC/qlqc_trainval_global_batch16/checkpoint_final.pth
```

## 3. 环境与预检

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

set -o pipefail
bash scripts/运行Car专项质量校准.sh preflight \
  2>&1 | tee Car专项质量校准预检.log
PRECHECK_EXIT=${PIPESTATUS[0]}
echo "预检退出码：$PRECHECK_EXIT"
```

只有退出码为 `0` 才继续。

## 4. 冒烟训练

```bash
set -o pipefail
bash scripts/运行Car专项质量校准.sh smoke \
  2>&1 | tee Car专项质量校准冒烟终端.log
SMOKE_EXIT=${PIPESTATUS[0]}
echo "冒烟退出码：$SMOKE_EXIT"
```

通过标准包括：

- 完成 2 个批次的前向、反向和优化更新
- 生成 epoch 为 1 的冒烟 checkpoint
- 权重审计显示只新增并改变 `car_quality_head.*`
- 新增参数数为 8257，所有旧模型张量逐位不变

## 5. 后台训练

```bash
screen -dmS car_quality bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行Car专项质量校准.sh train
'
```

查看进程与日志：

```bash
screen -ls
pgrep -af '训练与评估_无蒸馏QECR.py' \
  || echo "Car专项训练已经结束"

tr '\r' '\n' \
  < 'outputs/Car专项质量校准/T3_Car残差质量/控制台.log' \
  | tail -n 50

bash scripts/运行Car专项质量校准.sh status
```

脚本每轮保存滚动 checkpoint。正常中断后再次执行 `train` 会从已有滚动 checkpoint 继续，不会从零开始。完成标志为：

```text
完成  Car残差质量头3轮
```

## 6. KITTI 测试集推理和打包

训练完成后执行：

```bash
screen -dmS car_quality_test bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行Car专项质量校准.sh infer
'
```

完成后检查：

```bash
bash scripts/运行Car专项质量校准.sh status

ZIP_FILE='outputs/Car专项质量校准/QLQC_Car_residual_KITTI_test_submission.zip'
unzip -t "$ZIP_FILE" | tail -n 3
unzip -Z1 "$ZIP_FILE" | wc -l
cat "$ZIP_FILE.sha256"
```

文件数必须为 `7518`，ZIP 完整性测试必须无错误。最终上传 KITTI 官网的文件是：

```text
outputs/Car专项质量校准/QLQC_Car_residual_KITTI_test_submission.zip
```

## 7. 结果记录原则

官网恢复提交次数后只提交上述固定候选一次。需要完整记录 Car、Pedestrian、Cyclist 的 2D、AOS、3D 和 BEV 的 Easy、Moderate、Hard 结果，并与以下两组结果比较：

- StereoDETR 论文或官方榜单结果
- 同一训练环境下的全局批量 16 QLQC 结果

若新候选未提升 Car 3D AP，仍应如实保存结果。不能只选择有利难度或将无标签全量训练的官网结果表述为验证集消融。
