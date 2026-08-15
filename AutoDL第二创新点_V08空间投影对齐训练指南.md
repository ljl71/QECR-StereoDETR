# AutoDL 第二创新点：V08 空间—投影协同对齐训练指南

更新时间：2026-08-10

## 一、消融关系

V08O/A/B/C都不包含创新点1的V06B质量头，目的是先独立验证创新点2：

| 版本 | 结构 | 用途 |
|---|---|---|
| V00 | 官方StereoDETR | 195轮正式基线 |
| V08O | V00结构 + 从V00最佳权重低学习率续训5轮 | 公平控制额外续训、关闭增强和最佳权重选择造成的变化 |
| V08A | V08O协议 + 分阶段三维角点损失 | 角点约束独立消融 |
| V08B | V08O协议 + 分阶段左目投影损失 | 投影约束独立消融 |
| V08C | V08O协议 + 两种损失 | 第二创新点候选组合 |

V08O没有新增可学习模块，也没有创新点1。它与V00模型结构相同，但训练状态不同。只有V08A或V08B通过筛选后，才训练V08C；第二创新点最终通过后，再建立“V06B创新点1 + V08第二创新点”的组合版本。

## 二、统一输出目录

从V08开始，日志不再散落在`outputs`根目录：

```text
outputs/第二创新点_V08空间投影对齐/
├── 冒烟测试/
├── V08O_公平控制组/
│   ├── 控制台.log
│   ├── 配置快照.yaml
│   ├── 运行信息.txt
│   └── qecr_v08o_geometry_control/checkpoint_best.pth
├── V08A_三维角点对齐/
├── V08B_左目投影对齐/
└── V08C_空间投影协同对齐/
```

统一启动脚本会自动创建目录、保存配置快照、记录环境/时间/退出码，并拒绝覆盖已有日志。

## 三、上传后检查

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

bash scripts/运行QECR测试.sh
```

必须看到：

```text
20 个无蒸馏正式消融配置检查通过
V08空间角点、左目投影、分阶段权重与反向传播测试通过
QECR pre-training checks passed
```

## 四、整理已完成的V08O

只迁移V08O，不整理历史输出，避免破坏旧配置和checkpoint引用：

```bash
cd /root/autodl-tmp/QECR-StereoDETR

NEW_ROOT='outputs/第二创新点_V08空间投影对齐'
OLD_DIR='outputs/空间投影对齐短程筛选/V08O_几何对齐短程对照'
NEW_DIR="$NEW_ROOT/V08O_公平控制组"

mkdir -p "$NEW_ROOT"

if [ -d "$OLD_DIR" ] && [ ! -e "$NEW_DIR" ]; then
  mv "$OLD_DIR" "$NEW_DIR"
fi

if [ -f 'outputs/V08O_控制台.log' ] && [ ! -e "$NEW_DIR/控制台.log" ]; then
  mv 'outputs/V08O_控制台.log' "$NEW_DIR/控制台.log"
fi

find "$NEW_ROOT" -maxdepth 3 -type f -printf '%p  %k KB\n' | sort
```

## 五、运行V08A

进入screen：

```bash
screen -S v08a_corner
```

进入后只需：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open
bash scripts/运行V08短程实验.sh V08A
```

按`Ctrl+A`、再按`D`退出screen。日志统一位于：

```text
outputs/第二创新点_V08空间投影对齐/V08A_三维角点对齐/控制台.log
```

## 六、结果门槛

V08O最佳checkpoint为：

```text
Car 3D AP_R40 Easy/Moderate/Hard = 68.2163 / 49.5345 / 41.6798
Pedestrian/Cyclist Moderate = 24.4191 / 22.5256
```

V08A直接通过要求Moderate至少`49.7345`；若为`49.6345—49.7345`，还要求Hard、Pedestrian Moderate、Cyclist Moderate至少两项改善；Hard不得低于`41.5298`。V08A不通过仍可独立运行V08B，但不运行V08C和195轮。

## 七、后续统一命令

```bash
bash scripts/运行V08短程实验.sh V08B
bash scripts/运行V08短程实验.sh V08C
```

V08C只有在V08A或V08B至少一个通过后才能运行。每个版本都从同一V00 checkpoint开始，不串行继承V08O或前一版本权重。
