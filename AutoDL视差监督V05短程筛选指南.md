# AutoDL 视差监督 V05 短程筛选指南

## 一、结论和实验边界

200 样本正式 G0 得到：

- SGBM 内 16 候选相对单值的目标边界支持覆盖率提高 4.76 个百分点；
- 候选最小误差改善 1.649 px；
- 硬 LRC 会损失 8.50 个百分点有效率和 1.86 个百分点全点支持覆盖率；
- 因此只批准“训练期 SGBM + 多候选概率质量 + LRC 软权重”的 10 轮筛选，不批准直接跑 195 轮。

V05 不增加模型参数，不改变前向推理图。SGBM、候选整理和 LRC 只在训练数据加载与辅助视差损失中执行，因此正式模型推理延迟不增加；训练数据准备可能变慢。

## 二、四个短程版本

| 版本 | 匹配器 | 标签 | LRC | 作用 |
|---|---|---|---|---|
| V05A | BM | 最大池化单值 | 无 | 同起点、同学习率的续训对照 |
| V05B | SGBM | 最大池化单值 | 无 | 单独验证修复 SGBM 配置失效 |
| V05C | SGBM | 4×4 内16候选概率质量 | 无 | 单独验证核心多候选监督 |
| V05D | SGBM | 4×4 内16候选概率质量 | 软权重 | 完整候选监督版本 |

为节约费用，第一阶段只跑 V05A 和 V05D。只有 V05D 相对 V05A 通过门槛，才补跑 V05B、V05C 完成消融归因。

## 三、上传后检查

继续使用已有环境，不新建 Conda 环境，也不需要重新编译 CUDA 算子。

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

test -f \
  outputs/QECR消融实验/V00_官方基线/qecr_v00_official_baseline/checkpoint_best.pth \
  && echo "V00起点权重存在"

python - <<'PY'
import torch

path = (
    "outputs/QECR消融实验/V00_官方基线/"
    "qecr_v00_official_baseline/checkpoint_best.pth"
)
checkpoint = torch.load(path, map_location="cpu")
print("V00权重轮次:", checkpoint.get("epoch"))
print("V00记录最佳值:", checkpoint.get("best_result"))
print("参数数量:", len(checkpoint["model_state"]))
PY
```

然后运行代码门禁：

```bash
bash scripts/运行QECR测试.sh
echo "测试退出码：$?"
```

必须看到：

```text
多候选视差监督测试通过：SGBM生效、16候选不丢失、LRC为软权重、概率质量归一化且反向传播正常
StereoDETR PyTorch 2.0 完整模型导入链测试通过
QECR pre-training checks passed
```

## 四、V05D 一轮小样本冒烟

先创建固定的24样本列表：

```bash
head -n 24 \
  /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train.txt \
  > /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train_v05_smoke.txt

wc -l \
  /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train_v05_smoke.txt
```

运行：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
mkdir -p outputs/视差监督冒烟测试/V05D_一轮
set -o pipefail

python tools/训练与评估_无蒸馏QECR.py \
  --config configs/AutoDL_V05D_一轮冒烟.yaml \
  2>&1 | tee outputs/视差监督冒烟测试/V05D_一轮/控制台.log

echo "V05D冒烟退出码：${PIPESTATUS[0]}"
```

退出码必须为0，且至少完成一次前向、`loss_disp_map`、反向传播、保存和小样本评估。该 AP 没有统计意义。

## 五、第一阶段：先跑 V05A 对照

```bash
screen -S v05a_control
```

进入 screen 后：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p outputs/视差监督短程筛选/V05A_短程续训对照
set -o pipefail

python tools/训练与评估_无蒸馏QECR.py \
  --config versions/V05A_短程续训对照/config.yaml \
  2>&1 | tee outputs/视差监督短程筛选/V05A_短程续训对照/控制台.log
```

用 `Ctrl+A`、松开后按 `D` 脱离。

## 六、第一阶段：再跑 V05D 完整监督

V05A 完成后再启动，避免两个任务抢同一张4090：

```bash
screen -S v05d_full
```

进入 screen 后：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p outputs/视差监督短程筛选/V05D_多候选加LRC软权重
set -o pipefail

python tools/训练与评估_无蒸馏QECR.py \
  --config versions/V05D_多候选加LRC软权重/config.yaml \
  2>&1 | tee outputs/视差监督短程筛选/V05D_多候选加LRC软权重/控制台.log
```

## 七、比较门槛

分别查看：

```bash
grep "Best Result:" \
  outputs/视差监督短程筛选/V05A_短程续训对照/控制台.log \
  | tail -n 3

grep "Best Result:" \
  outputs/视差监督短程筛选/V05D_多候选加LRC软权重/控制台.log \
  | tail -n 3
```

第一判断指标是同一验证集上的 `Car 3D AP_R40 Moderate`，即日志中的 `Best Result`。筛选规则：

- V05D − V05A ≥ 0.30，且 Hard 不下降超过0.30：通过，继续 V05B/V05C 消融；
- 差值在 −0.10 到 +0.30：证据不足，先检查曲线和第二随机种子，不跑195轮；
- V05D − V05A < −0.10：停止该路线，不补跑完整消融。

不能直接用 V05D 的10轮结果减去历史 V00 的195轮结果来宣称增益。V05A 与 V05D 都从同一个 V00 最优权重开始、使用相同的10轮和 `2e-5` 学习率，这个配对差值才是主要因果证据。

## 八、通过后才补跑的 V05B/V05C

```bash
python tools/训练与评估_无蒸馏QECR.py \
  --config versions/V05B_SGBM单值修正/config.yaml

python tools/训练与评估_无蒸馏QECR.py \
  --config versions/V05C_SGBM多候选监督/config.yaml
```

消融解释：

- V05B − V05A：修复 SGBM 配置失效的作用；
- V05C − V05B：多候选概率质量的净作用；
- V05D − V05C：LRC 软置信度的净作用；
- V05D − V05A：完整训练期监督方案的总作用。

只有短程配对通过后，才讨论完整训练、第二随机种子和论文主创新点命名。
