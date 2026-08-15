# AutoDL 多候选视差监督 G0 诊断指南

## 现在先做什么

先不要训练 V04、V05，也不要再跑 195 轮。当前第一步是执行 **G0 离线诊断**，用 KITTI 稀疏 Velodyne 点回答三个问题：

1. StereoDETR 当前的 `StereoBM + 4×4 最大池化` 是否经常丢掉正确视差；
2. 每个 4×4 网格保留 16 个候选，是否明显提高目标和目标边界处的正确视差覆盖率；
3. SGBM 和左右一致性检查（LRC）在这套数据与参数上究竟是增益、无效还是会过度丢点。

这一步不训练网络，不改模型，也不把激光雷达用于训练。点云只作为离线评测尺子，因此后续模型仍然是纯双目视觉方法。

## 需要上传的文件

把本地最新项目同步到服务器 `/root/autodl-tmp/QECR-StereoDETR`。至少确认下面两个新文件存在：

```bash
cd /root/autodl-tmp/QECR-StereoDETR

ls -lh \
  tools/诊断多候选视差监督.py \
  AutoDL多候选视差监督诊断指南.md
```

不要覆盖或删除服务器已有的 `outputs/QECR消融实验`，PyCharm 部署应使用“上传已修改文件”，不要执行清空远端目录。

## 第一步：检查输入，不解压 28 GB 点云

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export OMP_NUM_THREADS=8

ls -lh \
  /autodl-pub/data/KITTI_Object/raw/data_object_velodyne.zip

wc -l \
  /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train.txt

for dir in image_2 image_3 calib label_2
do
  printf '%-10s ' "$dir"
  find "/root/autodl-tmp/datasets/KITTI/object/training/$dir" \
    -maxdepth 1 -type f | wc -l
done
```

预期：

- `train.txt` 为 3712 行；
- 四个目录各有 7481 个文件；
- Velodyne ZIP 存在即可。

**不要解压整个 `data_object_velodyne.zip`。** 脚本会直接从公共 ZIP 中按样本号读取约 200 个 `.bin`，不会额外占用几十 GB 数据盘。

## 第二步：运行不依赖数据的自检

```bash
python tools/诊断多候选视差监督.py --self_test
```

必须看到：

```text
G0 自检通过：网格候选、最大池化、LRC、投影和指标逻辑均正常
```

## 第三步：先做 20 样本真实冒烟诊断

```bash
mkdir -p outputs/视差监督诊断_G0_20样本冒烟

set -o pipefail

python tools/诊断多候选视差监督.py \
  --num_samples 20 \
  --output_dir outputs/视差监督诊断_G0_20样本冒烟 \
  2>&1 | tee outputs/视差监督诊断_G0_20样本冒烟/运行.log

G0_SMOKE_EXIT=${PIPESTATUS[0]}
echo "20样本诊断退出码：$G0_SMOKE_EXIT"
```

要求：

- 退出码为 `0`；
- 成功样本接近 20，不能全部失败；
- 输出目录出现 `诊断报告.md`、`诊断汇总.csv`、`诊断明细.json`。

检查：

```bash
ls -lh outputs/视差监督诊断_G0_20样本冒烟
sed -n '1,100p' outputs/视差监督诊断_G0_20样本冒烟/诊断报告.md
```

20 样本只检查管线，不能据此决定创新方向。

## 第四步：在 screen 中运行 200 样本正式 G0

如当前不在 screen：

```bash
screen -S qecr_g0
```

进入 screen 后执行：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export OMP_NUM_THREADS=8

mkdir -p outputs/视差监督诊断_G0
set -o pipefail

python tools/诊断多候选视差监督.py \
  --num_samples 200 \
  --seed 20260808 \
  --data_root /root/autodl-tmp/datasets/KITTI/object/training \
  --velodyne_zip /autodl-pub/data/KITTI_Object/raw/data_object_velodyne.zip \
  --output_dir outputs/视差监督诊断_G0 \
  2>&1 | tee outputs/视差监督诊断_G0/运行.log

G0_EXIT=${PIPESTATUS[0]}
echo "G0正式诊断退出码：$G0_EXIT"
```

用 `Ctrl+A`，松开后按 `D`，可安全退出 screen 而不中止脚本。

在普通终端检查：

```bash
screen -ls

tail -f \
  /root/autodl-tmp/QECR-StereoDETR/outputs/视差监督诊断_G0/运行.log
```

该脚本主要使用 CPU，GPU 利用率低是正常现象。耗时受 AutoDL 公共存储读取速度影响，不要因为短时间停在某个样本就立即中止；日志每 10 个样本会打印一次进度。

## 第五步：完成后把这些结果发回来

```bash
cd /root/autodl-tmp/QECR-StereoDETR

cat outputs/视差监督诊断_G0/诊断报告.md

python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/视差监督诊断_G0/诊断明细.json")
data = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(data["自动判断"], ensure_ascii=False, indent=2))
print("成功样本:", data["元数据"]["实际完成样本数"])
print("失败样本:", len(data["元数据"]["失败样本"]))
print("投影点数:", data["元数据"]["有效投影点总数"])
PY
```

把 `诊断报告.md` 的全文和上面 JSON 摘要发给我。随后按证据走下面三种分支：

- **多候选通过**：我再实现正式 V05-MC-PMC，只改训练期监督与损失，推理期不新增分支，然后先做 10–15 轮配对消融；
- **多候选谨慎**：只实现最小版本，先短训，不跑完整 195 轮；
- **多候选否决**：停止这条路线，不继续消耗服务器费用。

## 为什么这一步比直接训练更划算

当前 V01、V02、V03 已连续低于 V00，说明继续凭直觉堆查询模块的成功概率不够高。G0 只需要处理少量样本，却能在写正式训练代码前验证“监督标签是否确实存在可修复的信息损失”。它不能保证 AP 提升，但能以较低成本排除没有数据依据的方向。

## 指标解释

- **单值支持覆盖率**：最大池化得到的唯一标签在 ±2 px 内接近激光参照的比例；
- **多候选支持覆盖率**：16 个候选中至少有一个在 ±2 px 内的比例；
- **候选最小误差**：集合中最接近参照的候选误差，是 oracle 上界，不是模型 EPE；
- **有效率**：该方法在参照点所在 4×4 网格中至少提供一个有效候选的比例；
- **目标边界**：KITTI 2D 框边缘 8 px 内的近似区域，不等同于真实实例轮廓；
- **LRC**：左视差与右视差回查是否一致。若它能降误差但大量丢点，正式实现只应采用软权重，不能硬过滤。
