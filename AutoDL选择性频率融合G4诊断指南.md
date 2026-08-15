# AutoDL 选择性频率融合 G4 诊断指南

## 这一步做什么

G4 不是训练，也不会改 V09 权重。它用冻结 V09 检查：

- 当前 `s4` 原始相关是否适合目标边界等高频区域；
- `3×3` 局部平滑相关是否适合平坦、弱纹理等低频区域；
- 左目上下文边缘强度能否在独立留出样本上预测应选哪一路；
- Pedestrian/Cyclist 是否会被该选择策略伤害。

只有自动结论为“通过，允许实现V13配对短训”，本地才会修改正式模型。其他结论均停止该路线。

## 一、确认文件已上传

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

ls -lh \
  tools/诊断选择性频率融合_G4.py \
  AutoDL选择性频率融合G4诊断指南.md
```

## 二、环境与自检

```bash
export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

python tools/诊断选择性频率融合_G4.py --self_test
```

预期输出：

```text
G4自检通过：稀疏双频曲线、样本外阈值门控和指标汇总逻辑正常
```

## 三、确认 V09 与点云压缩包

```bash
V09_CKPT='outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth'
VELODYNE_ZIP='/autodl-pub/data/KITTI_Object/raw/data_object_velodyne.zip'

ls -lh "$V09_CKPT" "$VELODYNE_ZIP"
sha256sum "$V09_CKPT"
```

如果平台公共数据实际挂载在 `/root/autodl-pub`，先执行：

```bash
find /root/autodl-pub /autodl-pub/data \
  -maxdepth 5 \
  -type f \
  -name 'data_object_velodyne.zip' \
  -print 2>/dev/null
```

然后把下方 `--velodyne_zip` 改成查到的真实路径。

## 四、先跑 20 样本冒烟

这一步只检查完整推理、点云投影、样本划分和报告生成，不据此决定模块是否有效。

```bash
mkdir -p 'outputs/选择性频率融合诊断_G4/G4_20样本冒烟'
set -o pipefail

python tools/诊断选择性频率融合_G4.py \
  --checkpoint "$V09_CKPT" \
  --velodyne_zip "$VELODYNE_ZIP" \
  --num_samples 20 \
  --workers 4 \
  --output_dir 'outputs/选择性频率融合诊断_G4/G4_20样本冒烟' \
  2>&1 | tee 'outputs/选择性频率融合诊断_G4/G4_20样本冒烟/运行.log'

G4_SMOKE_EXIT=${PIPESTATUS[0]}
echo "G4冒烟退出码：$G4_SMOKE_EXIT"
```

`20` 样本低于预注册的 `80` 样本下限，因此即使程序正常，自动结论也应是“证据不足”或“否决”，不能用它决定正式改码。

检查：

```bash
cat 'outputs/选择性频率融合诊断_G4/G4_20样本冒烟/诊断报告.md'
```

## 五、正式运行 100 样本诊断

确认冒烟退出码为 `0` 后运行：

```bash
mkdir -p 'outputs/选择性频率融合诊断_G4/G4_100样本'
set -o pipefail

python tools/诊断选择性频率融合_G4.py \
  --checkpoint "$V09_CKPT" \
  --velodyne_zip "$VELODYNE_ZIP" \
  --num_samples 100 \
  --seed 20260812 \
  --points_per_region 384 \
  --workers 4 \
  --output_dir 'outputs/选择性频率融合诊断_G4/G4_100样本' \
  2>&1 | tee 'outputs/选择性频率融合诊断_G4/G4_100样本/运行.log'

G4_EXIT=${PIPESTATUS[0]}
echo "G4正式诊断退出码：$G4_EXIT"
```

该过程只读 V09 和 Velodyne ZIP，不会完整解压点云，也不会保存新的模型权重。

## 六、完成后发回哪些信息

```bash
cat 'outputs/选择性频率融合诊断_G4/G4_100样本/诊断报告.md'

echo '===== 退出与异常检查 ====='
grep -E \
  '自动结论：|解释：|Traceback|CUDA out of memory|所有样本|没有可合并' \
  'outputs/选择性频率融合诊断_G4/G4_100样本/运行.log' \
  || true

echo '===== 产物 ====='
find 'outputs/选择性频率融合诊断_G4/G4_100样本' \
  -maxdepth 1 -type f -printf '%f  %k KB\n'
```

把以上完整输出发回即可，不需要先跑任何训练。

## 七、如何理解四种结果

- `当前原始相关`：V09 当前 `s4` 相关曲线，是比较基准。
- `固定局部平滑`：所有位置都使用低频代理，用来判断“固定平滑”是否会伤边缘。
- `双分支_Oracle`：使用真值逐点选最好的一路，只是信息上限。
- `上下文阈值门控`：只用校准样本确定阈值，再在未见留出样本上执行，是唯一接近可实现模块的诊断结果。

正式通过必须同时满足：

1. Oracle 对目标点和 Car 都有足够上限；
2. 上下文门控在留出集兑现至少一部分上限；
3. Pedestrian/Cyclist 不越过保护线；
4. 目标边界比目标内部明显更常选择原始高频分支；
5. 成功样本和稀疏点数量充分。

如果只满足第 1 条而第 2 条失败，仍然停止。这样可以避免为了追逐 Oracle 而加入昂贵、不可泛化的复杂门控。
