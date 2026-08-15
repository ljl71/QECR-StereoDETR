# AutoDL 无蒸馏 QECR-StereoDETR 训练与实验指南

## 1. 使用哪个工程和入口

工程目录：

```text
code/ablation_versions/QECR-StereoDETR
```

唯一正式训练入口：

```bash
python tools/训练与评估_无蒸馏QECR.py --config <配置文件>
```

这个入口会优先安装经过本地前向、反向传播和越界回退测试的 `query_epipolar_runtime.py`。不要直接运行开发中间入口 `训练与评估_QECR.py`。

本工程不需要 Fast-FoundationStereo 环境、教师权重或教师缓存。

## 2. AutoDL 机器建议

优先选择单张 RTX 4090 24 GB、RTX A5000 24 GB、RTX A6000 48 GB 或同级 GPU。所有消融必须使用同一型号、同一输入尺寸、同一 batch size 和同一精度设置。

RTX 4090 是 Ada 架构，计算能力为 8.9。CUDA 11.8 起可以原生生成 `sm_89` cubin。因此选择 4090 时，应使用包含 CUDA 11.8 或更新版 `nvcc` 的开发镜像，不要继续使用官方旧 `setup.py` 中只编译到 `sm_75` 的参数。参考：

- PyTorch CUDA 扩展与 `TORCH_CUDA_ARCH_LIST`：<https://docs.pytorch.org/docs/main/cpp_extension.html>
- NVIDIA Ada 兼容指南：<https://docs.nvidia.com/cuda/ada-compatibility-guide/>

## 3. 本轮正式环境

QECR 消融必须复用已经完成 V00 基线训练的环境，不新建 Conda 环境：

```text
Conda：/root/miniconda3/envs/stereodetr-open
Python：3.8.20
PyTorch：2.0.1+cu118
torchvision：0.15.2+cu118
CUDA Toolkit / nvcc：11.8
GPU：RTX 4090 24 GB
```

在 PyCharm 的 SSH 项目向导中选择“选择现有”，解释器填写：

```text
/root/miniconda3/envs/stereodetr-open/bin/python
```

远端同步目录使用：

```text
/root/autodl-tmp/QECR-StereoDETR
```

激活并检查：

```bash
source /root/miniconda3/bin/activate stereodetr-open

python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY

nvcc --version
```

不要为 QECR 另装 PyTorch 2.4 或 CUDA 12 环境，否则环境差异会污染与 V00 基线的公平比较。

## 4. 数据目录

默认配置假设：

```text
/root/autodl-tmp/datasets/KITTI/object/training/
├── ImageSets/
│   ├── train.txt
│   └── val.txt
├── calib/
├── image_2/
├── image_3/
└── label_2/
```

检查：

```bash
ls /root/autodl-tmp/datasets/KITTI/object/training/image_2 | head
ls /root/autodl-tmp/datasets/KITTI/object/training/image_3 | head
ls /root/autodl-tmp/datasets/KITTI/object/training/calib | head
```

如果平台实际路径不同，只修改：

```text
configs/AutoDL无蒸馏基础配置.yaml
```

中的 `root_dir`、`train_txt`、`root_dir_eval` 和 `eval_txt`。

## 5. 编译 4090 可用的 CUDA 算子

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export TORCH_CUDA_ARCH_LIST="8.9"
bash scripts/编译CUDA算子_QECR.sh
```

3090/A5000 通常使用计算能力 8.6：

```bash
export TORCH_CUDA_ARCH_LIST="8.6"
bash scripts/编译CUDA算子_QECR.sh
```

必须在实际租用的 GPU 实例内重新编译，不要上传其他机器的 `build`、`.so` 或 `__pycache__`。

## 6. 训练前测试

```bash
bash scripts/运行QECR测试.sh
```

它会检查：

1. 九个消融配置没有任何蒸馏开关；
2. V03/V04 的安全融合初始查询权重为0.05且不会超过0.50；
3. 查询深度与极线模块概率和为 1；
4. 查询深度、基线深度和左右采样均能反向传播；
5. 越界匹配会自动回退到原查询分布；
6. P2/P3 基线、随机裁剪深度尺度和翻转方向正确；
7. 三层输出、双向一致性和几何门控可以组合运行；
8. Python 文件无语法错误。

任何一项失败都不要开始 195 epoch 正式训练。

## 7. 先做小样本冒烟训练

```bash
head -n 100 \
  /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train.txt \
  > /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/smoke_train.txt

head -n 50 \
  /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/val.txt \
  > /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/smoke_val.txt
```

临时复制 V03 配置，把 `train_txt/eval_txt` 指向上述文件、`batch_size` 改为 2、`trainer.max_epoch` 改为 2。确认：

- loss 有限且能下降；
- `loss_query_dist` 存在且有限；
- 输出包含 `pred_baseline_depth`、`pred_raw_query_depth` 和 `pred_query_depth_blend_weight`；
- 没有 `loss_teacher_distill`；
- GPU 显存稳定；
- 保存和验证均能完成。

## 8. 当前修正版训练顺序

统一先设置：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open
export CUDA_VISIBLE_DEVICES=0
```

V00、V01、V02 已完成。当前只依次运行 V03、V04：

```bash
python tools/训练与评估_无蒸馏QECR.py \
  --config versions/V03_基线安全查询深度/config.yaml

python tools/训练与评估_无蒸馏QECR.py \
  --config versions/V04_基线安全跨层细化/config.yaml
```

V03 只检验基线安全写回；V04 只比 V03 多跨层细化。V03 未完成前不要启动 V04。旧 V15—V18 仍继承负增益 V02，暂不运行；只有 V03/V04 达到恢复门槛后才迁移极线模块。

## 9. 显存不足

本轮所有版本固定使用 `batch_size=12`，与已经完成的 V00 本地基线一致。官方 Batch 16 已在当前 RTX 4090、PyTorch 2.0.1 FP32 环境中确认 OOM。

若某个改进版本在 Batch 12 下 OOM，不允许只降低该版本的 Batch Size 后直接与 V00 比较。应先分析新增模块的显存占用；若必须统一降低 Batch Size，需要重新建立同 Batch Size 的 V00 对照。

## 10. 评估和时延

评估：

```bash
python tools/训练与评估_无蒸馏QECR.py -e \
  --config versions/V03_基线安全查询深度/config.yaml
```

训练 V04 时将上述配置替换为 `versions/V04_基线安全跨层细化/config.yaml`。

同步 CUDA 的单图时延：

```bash
python tools/测量QECR端到端时延.py \
  --config versions/V03_基线安全查询深度/config.yaml \
  --checkpoint outputs/QECR消融实验/V03_基线安全查询深度/qecr_v03_baseline_safe_query_depth/checkpoint_best.pth
```

不要使用原 `tester_helper.py` 中未经 `torch.cuda.synchronize()` 的 `time.time()` 结果作为论文时延。

## 11. 必须保存的实验结果

每个版本至少保存：

- Car/Pedestrian/Cyclist 的 3D AP R40；
- BEV AP R40；
- Easy/Moderate/Hard；
- 最佳 epoch；
- batch size、GPU、PyTorch、CUDA、随机种子；
- 参数量、峰值显存、median/P90 时延；
- `pred_epipolar_confidence`、`pred_epipolar_cycle_error` 的统计；
- 按遮挡、截断和距离分组的结果。

只有 V15 相对 V02 超过实验波动，才能说局部极线校正有效；只有 V16 相对 V15 有稳定增益，才能说双向一致性有效。
