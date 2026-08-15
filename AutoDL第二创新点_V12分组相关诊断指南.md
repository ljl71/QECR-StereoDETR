# AutoDL 第二创新点 V12 分组相关诊断指南

## 这次先做什么

本次上传的是 V12-G0 证据诊断，不会修改 V09 权重，不会训练新模型，也不会完整解压 Velodyne。先用 20 个样本检查流程，再用固定随机种子的 100 个样本决定是否值得开发正式 V12A。

请先把本地 `QECR-StereoDETR` 更新内容同步到：

```text
/root/autodl-tmp/QECR-StereoDETR
```

不要上传整个 `code/stereo_matching_modules`。那些官方仓库是本地论文和实现审计依据，AutoDL 运行 G0 不需要它们。

## 一、进入环境并检查文件

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

ls -lh \
  tools/诊断分组相关信息_V12G0.py \
  scripts/运行V12分组相关诊断.sh \
  第二创新点_V12分组相关信息保留路线论证.md
```

确认 V09 权重仍在：

```bash
V09_CKPT='outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth'

ls -lh "$V09_CKPT"
sha256sum "$V09_CKPT"
```

正确 SHA256 应为：

```text
6b673fef2a54e2a5fbc44be731b24914770fffebaf1c20a0cacc43228dd316b4
```

确认点云 ZIP 存在：

```bash
ls -lh /autodl-pub/data/KITTI_Object/raw/data_object_velodyne.zip
```

## 二、运行训练前总测试

```bash
set -o pipefail
bash scripts/运行QECR测试.sh 2>&1 | tee V12训练前测试.log
TEST_EXIT=${PIPESTATUS[0]}
echo "测试退出码：$TEST_EXIT"
```

应看到：

```text
V12-G0自检通过：分组相关、均匀等价、Oracle指标和止损门槛正常
QECR pre-training checks passed
测试退出码：0
```

## 三、20 样本冒烟

```bash
bash scripts/运行V12分组相关诊断.sh smoke
```

输出统一保存在：

```text
outputs/第二创新点_V12分组相关诊断/G0_20样本冒烟/
├── V09配置快照.yaml
├── 运行信息.txt
├── 控制台.log
├── 诊断报告.md
├── 诊断汇总.csv
└── 诊断明细.json
```

20 个样本只用于检查程序和初步趋势，报告中的自动状态应是“证据不足”，不能据此决定实现模块。

查看关键信息：

```bash
cat 'outputs/第二创新点_V12分组相关诊断/G0_20样本冒烟/诊断报告.md'
```

若退出码为 0、失败样本很少且均匀平均等价误差接近 `1e-6` 或更小，再进行正式诊断。

## 四、100 样本正式诊断

```bash
bash scripts/运行V12分组相关诊断.sh formal
```

输出保存在：

```text
outputs/第二创新点_V12分组相关诊断/G0_100样本正式诊断/
```

结束后执行：

```bash
REPORT='outputs/第二创新点_V12分组相关诊断/G0_100样本正式诊断/诊断报告.md'
cat "$REPORT"
```

再把下面命令的输出发回来：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('outputs/第二创新点_V12分组相关诊断/G0_100样本正式诊断/诊断明细.json')
data = json.loads(path.read_text(encoding='utf-8'))
print('自动判断：')
print(json.dumps(data['判断'], ensure_ascii=False, indent=2))
print('\n元数据：')
print(json.dumps(data['元数据'], ensure_ascii=False, indent=2))
PY
```

## 五、如何解释结果

- `通过进入轻量门控短程配对`：只说明分组中存在显著可利用信息。接下来才修改主模型，实现零初始化门控，并跑 V12O/V12A 的 5 轮配对。
- `证据不足`：通常是样本或 Car 激光点不够；不要自行放宽门槛，先检查失败样本和点数。
- `否决`：停止 V12，不实现门控，不跑完整训练。

无论哪种结果，`逐点最优分组_Oracle` 都使用了真实视差，不能写成模型 AP，也不能与 V09 的 49.8210 直接比较。

## 六、避免重复运行覆盖

脚本发现目标输出目录已有文件会主动停止。若需要复核，请保留旧目录并使用新的 `--output_dir` 手动运行；不要删除已经用于论文审计的报告。

