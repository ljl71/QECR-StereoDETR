# AutoDL V06B正式时延测试指南

## 1. 测量口径

本测试比较同一RTX 4090上的V00和最终V06B，使用：

- batch size 1；
- CUDA Event计时与每次同步；
- 每次50步预热、200步正式测量；
- V00/V06B交替，各重复3次；
- 相同数据、分辨率、环境和进程状态。

脚本测量的是模型前向，不含磁盘读取、CPU数据准备、KITTI文本写入和官方AP计算，因此论文中应写“同步模型前向时延”，不能写完整系统端到端时延。

## 2. 上传与检查

上传更新后的：

```text
tools/测量QECR端到端时延.py
```

确认没有训练占用GPU：

```bash
pgrep -af '训练与评估_无蒸馏QECR.py' || true
nvidia-smi
```

## 3. 正式配对测试

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

V00_CKPT='/root/autodl-tmp/QECR-StereoDETR/outputs/QECR消融实验/V00_官方基线/qecr_v00_official_baseline/checkpoint_best.pth'
V06B_CKPT='/root/autodl-tmp/QECR-StereoDETR/outputs/三维质量排序短程筛选/V06B_点式三维质量排序/qecr_v06b_pointwise_3d_quality/checkpoint_best.pth'

mkdir -p 'outputs/V06B正式时延测试'
set -o pipefail

for run_id in 1 2 3
do
  echo "===== 第${run_id}组：V00 ====="
  python tools/测量QECR端到端时延.py \
    --config 'versions/V00_官方基线/config.yaml' \
    --checkpoint "$V00_CKPT" \
    --warmup 50 \
    --steps 200 \
    2>&1 | tee "outputs/V06B正式时延测试/V00_第${run_id}次.log"
  test ${PIPESTATUS[0]} -eq 0 || exit 1

  echo "===== 第${run_id}组：V06B ====="
  python tools/测量QECR端到端时延.py \
    --config 'versions/V06B_点式三维质量排序/config.yaml' \
    --checkpoint "$V06B_CKPT" \
    --quality_score_power 1.5 \
    --warmup 50 \
    --steps 200 \
    2>&1 | tee "outputs/V06B正式时延测试/V06B_第${run_id}次.log"
  test ${PIPESTATUS[0]} -eq 0 || exit 1
done

echo "V00/V06B三组配对时延测试完成"
```

## 4. 汇总

```bash
grep -HE \
  'GPU:|parameters:|median_ms:|mean_ms:|p90_ms:|fps_from_median:' \
  outputs/V06B正式时延测试/*.log
```

把汇总结果发回本任务。最终比较使用3次median的平均值，同时报告波动范围、参数增量、相对时延百分比和FPS；如果单次异常，不私自删除，先检查当时GPU是否被其他进程占用。
