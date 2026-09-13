# AutoDL StereoDETR同环境基线训练与KITTI官网提交指南

## 目的

本流程生成与QLQC最终模型处于同一代码、数据和运行环境下的纯StereoDETR基线，用于论文中的公平对照。

- 训练数据：KITTI trainval，共7481组有标注双目图像
- 测试数据：KITTI test，共7518组无标注双目图像
- 训练轮数：固定195轮
- Batch size：12
- 关闭QLQC、GPSD以及其他QECR实验模块
- 不执行V09T1强控制和V09T2质量校准
- 不使用KITTI test结果选模型或调参数

如果服务器已经保留以下权重，脚本会跳过训练，直接进行测试集推理：

```text
outputs/KITTI_trainval官网复核/V09T0_trainval基础模型/
qecr_v09t0_trainval_stereodetr/checkpoint_final.pth
```

## 1. 进入环境

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
```

## 2. 上传后预检

```bash
set -o pipefail

bash scripts/运行StereoDETR同环境基线官网提交.sh preflight \
  2>&1 | tee StereoDETR同环境基线预检.log

PRECHECK_EXIT=${PIPESTATUS[0]}
echo "预检退出码：$PRECHECK_EXIT"
```

只有退出码为0才能继续。

## 3. 先查看已有权重

```bash
bash scripts/运行StereoDETR同环境基线官网提交.sh status
```

如果显示“训练完成”，不需要重新训练，直接执行：

```bash
bash scripts/运行StereoDETR同环境基线官网提交.sh infer
```

## 4. 缺少权重时训练并推理

建议在screen后台运行完整流程：

```bash
screen -dmS stereodetr_reimpl bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行StereoDETR同环境基线官网提交.sh all
'
```

`all`会依次执行预检、训练、测试集推理、结果格式审计和ZIP打包。若检测到合法的195轮固定权重，会自动跳过训练。若训练中断，重新运行同一命令会读取`checkpoint.pth`继续训练。

## 5. 查看进度

```bash
screen -ls

pgrep -af '训练与评估_无蒸馏QECR.py' \
  || echo "当前没有训练或推理进程"

tail -n 60 \
  'outputs/KITTI_trainval官网复核/V09T0_trainval基础模型/控制台.log'
```

测试集推理阶段查看：

```bash
tail -n 60 \
  'outputs/KITTI_trainval官网复核/官网测试/StereoDETR同环境基线/控制台.log'
```

统一查看状态：

```bash
bash scripts/运行StereoDETR同环境基线官网提交.sh status
```

## 6. 最终提交文件

成功后只提交下面这个ZIP：

```text
outputs/KITTI_trainval官网复核/
StereoDETR_Reimpl_trainval_KITTI_test_submission.zip
```

提交前可再次检查：

```bash
ZIP_FILE='outputs/KITTI_trainval官网复核/StereoDETR_Reimpl_trainval_KITTI_test_submission.zip'

unzip -t "$ZIP_FILE" | tail -n 3
unzip -Z1 "$ZIP_FILE" | wc -l
unzip -Z1 "$ZIP_FILE" | head -n 3
unzip -Z1 "$ZIP_FILE" | tail -n 3
cat "$ZIP_FILE.sha256"
```

应满足：

- ZIP内部正好7518个txt
- 文件从`000000.txt`连续到`007517.txt`
- txt直接位于ZIP根目录
- `unzip -t`无错误

## 7. 结果解释

官网返回结果后，主要记录Car、Pedestrian和Cyclist的3D Detection Easy、Moderate和Hard。该结果在论文中命名为`StereoDETR-Reimpl`，与QLQC-StereoDETR构成同环境公平比较。原论文公开的StereoDETR结果仍单独标记为`StereoDETR (published)`，不得与复现基线混写。
