# AutoDL三维质量分数融合G2诊断指南

## 1. 目的

V06B在冻结V00检测器的条件下，仅凭点式3D IoU质量头把Car 3D AP_R40 Moderate从`48.3226`提高到`48.4916`。V06C加入成对排序后只有`48.4924`，净变化`+0.0008 AP`，因此成对损失已被消融否决。

G2不训练、不修改checkpoint，只扫描最终置信度中的质量指数：

```text
最终分数 = 分类分数 × exp(-sigma) × 质量概率^beta
```

- `beta=0`：关闭质量因子，应恢复V00，用于检查评估路径；
- `beta=1`：精确复现当前V06B；
- `0<beta<1`：减弱质量头影响；
- `beta>1`：增强质量头影响。

该扫描是分数校准，不是新网络模块。所有beta共用同一个V06B checkpoint。

## 2. 上传文件

用PyCharm只上传以下5个已修改文件，保持相同相对路径：

```text
lib/models/monodetr/quality_ranking.py
lib/models/monodetr/stereodetr.py
configs/QECR默认配置.yaml
tools/训练与评估_QECR.py
tools/测试3D质量排序.py
```

不要上传本地checkpoint、outputs、build、dist或`.so`。

## 3. 上传后测试

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

bash scripts/运行QECR测试.sh
echo "测试退出码：$?"
```

必须看到3D质量排序测试通过和最终退出码0。

## 4. 一次运行全部beta

```bash
screen -S stereodetr_g2
```

进入screen后：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

mkdir -p 'outputs/三维质量排序融合诊断_G2'
set -o pipefail

for beta in 0 0.25 0.5 0.75 1 1.25 1.5 2
do
  tag=${beta/./p}
  echo "===== beta=$beta ====="

  python tools/训练与评估_无蒸馏QECR.py \
    --evaluate_only \
    --config 'versions/V06B_点式三维质量排序/config.yaml' \
    --quality_score_power "$beta" \
    2>&1 | tee \
    "outputs/三维质量排序融合诊断_G2/beta_${tag}.log"

  run_code=${PIPESTATUS[0]}
  if [ "$run_code" -ne 0 ]; then
    echo "beta=$beta 失败，退出码=$run_code"
    exit "$run_code"
  fi
done

echo "G2全部指数评估完成"
```

按`Ctrl+A`再按`D`退出screen。预计总耗时约20～25分钟。

## 5. 汇总结果

```bash
for log_file in outputs/三维质量排序融合诊断_G2/beta_*.log
do
  echo "===== $(basename "$log_file") ====="
  awk '
    /Car AP_R40@0.70, 0.70, 0.70:/ {inside=1; next}
    inside && /3d   AP:/ {print; exit}
  ' "$log_file"
done
```

其中每行3个数依次是Easy、Moderate、Hard。把输出发回本任务再决定beta。

## 6. 预先固定的判断规则

- `beta=0`应恢复V00的`48.3226`；偏差超过0.01时先排查路径，不选择beta。
- `beta=1`应复现V06B的`48.4916`；偏差超过0.01时先排查checkpoint或配置。
- 最佳beta相对`beta=1`再提高至少`0.10 AP`，且Hard不下降超过0.10：保留校准指数。
- 改善不足`0.10 AP`：固定`beta=1`，不把指数调节包装成创新点。
- beta确定后再对V00和最终V06B做CUDA同步、预热、多次重复的正式时延测试。

beta是在Chen验证集上选择的超参数，最终论文仍需在KITTI官方测试集或独立协议上报告泛化结果，不能把扫描中的验证集最优值描述为无偏测试结果。
