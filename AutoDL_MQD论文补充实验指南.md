# AutoDL：MQD 论文补充实验指南

## 1. 上传范围

可以直接用 PyCharm 把本地项目

`G:\ljlstudy\Stereo_3D_Object_Detection\code\ablation_versions\QECR-StereoDETR`

覆盖上传到服务器

`/root/autodl-tmp/QECR-StereoDETR`

本次改动不会改变现有 V09、Batch16 QLQC 和 Car 最终配置的默认推理行为。新增开关只由 `configs/MQD论文补充实验/` 中的配置启用。

## 2. 环境准备

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
```

## 3. 预检与冒烟

先运行只读预检：

```bash
set -o pipefail
bash scripts/运行MQD论文补充实验.sh preflight \
  2>&1 | tee MQD论文补充实验_预检.log
echo "预检退出码：${PIPESTATUS[0]}"
```

退出码为 0 后运行两批冒烟：

```bash
set -o pipefail
bash scripts/运行MQD论文补充实验.sh smoke \
  2>&1 | tee MQD论文补充实验_冒烟终端.log
echo "冒烟退出码：${PIPESTATUS[0]}"
```

冒烟只验证前向、反向、优化与固定范围开关，不把小样本 AP 写入论文。

## 4. 正式运行顺序

建议分阶段运行，便于发现问题后及时停止租卡。

### 4.1 CAGS 监督消融

```bash
screen -dmS mqd_shared bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行MQD论文补充实验.sh shared
'
```

该阶段依次训练 S1、S2、S3，每个版本从同一个 V08O checkpoint 出发，只训练共享 QCMR 参数。

### 4.2 BCDF 内部消融

```bash
screen -dmS mqd_boundary bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行MQD论文补充实验.sh boundary
'
```

该阶段依次训练 B1、B2、B3、B4，每个版本从同一个 V09 checkpoint 出发，只训练 `car_quality_head.*`。

### 4.3 完整 BCDF 第二随机种子

```bash
screen -dmS mqd_seed445 bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行MQD论文补充实验.sh seed445
'
```

### 4.4 度量一致性与时延

B4 完成后运行：

```bash
bash scripts/运行MQD论文补充实验.sh analysis
bash scripts/运行MQD论文补充实验.sh latency
bash scripts/运行MQD论文补充实验.sh summary
```

`analysis` 会逐项审计三个版本的候选类别、完整类别概率向量、深度可靠性、二维框编码、深度、三维尺寸、朝向和三维 IoU 目标保持一致。任何候选变化都会直接报错，避免把检测器变化误写成决策机制收益。

## 5. 进度查看

```bash
screen -ls
pgrep -af '训练与评估_无蒸馏QECR.py' || echo '当前没有训练进程'
bash scripts/运行MQD论文补充实验.sh status
```

查看某一阶段日志：

```bash
tail -n 60 'outputs/MQD论文补充实验/S1_Hungarian匹配查询监督/控制台.log'
tail -n 60 'outputs/MQD论文补充实验/B4_完整边界决策场/控制台.log'
```

需要连续观察进度条时：

```bash
tr '\r' '\n' \
  < 'outputs/MQD论文补充实验/B4_完整边界决策场/控制台.log' \
  | tail -n 40
```

## 6. 完成后需要发回的文件

```bash
cat 'outputs/MQD论文补充实验/MQD论文补充实验汇总.md'
cat 'outputs/MQD论文补充实验/度量一致性分析/度量一致性报告.md'
cat 'outputs/MQD论文补充实验/完整时延/完整时延报告.md'
```

另外保留：

- `outputs/MQD论文补充实验/度量一致性分析/度量一致性汇总.json`
- `outputs/MQD论文补充实验/度量一致性分析/查询级度量证据明细.csv`
- `outputs/MQD论文补充实验/度量一致性分析/定性案例索引.csv`
- `outputs/MQD论文补充实验/度量一致性分析/分数区间与三维IoU关系.png`
- 各版本的 `训练后参数审计.txt`

## 7. 结果采用规则

1. S1/S2/S3 只用于决定 CAGS 的监督覆盖与尾部平衡是否能够单独声明。
2. B1/B2/B3/B4 必须来自同一 V09 起点，参数审计必须通过。
3. B4 若只在一个随机种子上有效，不写成稳定独立贡献。
4. 机制指标只用于解释，不替代 KITTI AP。
5. 补充实验不生成 KITTI test 提交包，不消耗官网次数。
