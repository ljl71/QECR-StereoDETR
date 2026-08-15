# AutoDL V09强基线加三维质量排序实验指南

## 1. 本实验验证什么

V09不是继续尝试已经否决的V08几何损失，而是验证两个已经取得正结果的部分能否叠加：

- R0：复评V08O最佳checkpoint，直接分母应接近Car 3D AP_R40 Moderate `49.5345`。
- R1：使用同一个V08O权重，冻结整个StereoDETR检测器，只训练V06B点式三维质量头。
- 推理固定使用`beta=1.5`，不根据V09结果重新扫描指数。

V09新增训练参数只有质量头的`66,561`个参数。检测框本身由V08O冻结检测器产生，R1与R0的差异来自查询级三维质量置信度校准。

## 2. 上传时必须保留的服务器文件

PyCharm重新同步本地代码时，不要启用“删除服务器中本地不存在的文件”。服务器上的下面这个权重必须保留：

```text
/root/autodl-tmp/QECR-StereoDETR/outputs/第二创新点_V08空间投影对齐/V08O_公平控制组/qecr_v08o_geometry_control/checkpoint_best.pth
```

同步完成后先检查：

```bash
cd /root/autodl-tmp/QECR-StereoDETR

ls -lh \
  'outputs/第二创新点_V08空间投影对齐/V08O_公平控制组/qecr_v08o_geometry_control/checkpoint_best.pth'

ls -lh \
  'versions/V09_V08O加点式三维质量排序/config.yaml' \
  'scripts/运行V09强基线质量组合.sh'
```

## 3. 训练前完整检查

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

必须看到：

```text
checkpoint兼容白名单测试通过：只允许quality_head.*缺失
V09结果解析与止损判定自检通过
QECR pre-training checks passed
测试退出码：0
```

## 4. 一条命令完成R0、R1和结果汇总

建议在`screen`中运行：

```bash
screen -S v09_v08o_quality
```

进入screen后执行：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

bash scripts/运行V09强基线质量组合.sh all
```

按`Ctrl+A`，再按`D`退出screen而不中断实验。

重新查看：

```bash
screen -r v09_v08o_quality
```

## 5. 输出目录

```text
outputs/最终组合_V09/
├── R0_V08O直接复评/
│   ├── 控制台.log
│   ├── 配置快照.yaml
│   └── 运行信息.txt
├── V09_V08O加点式三维质量排序/
│   ├── 控制台.log
│   ├── 配置快照.yaml
│   ├── 运行信息.txt
│   └── qecr_v09_v08o_pointwise_3d_quality/
│       ├── checkpoint.pth
│       ├── checkpoint_best.pth
│       └── outputs/
└── V09结果对照.md
```

脚本会记录V08O checkpoint的SHA256，防止R0和R1误用不同权重；加载R1时只允许`quality_head.*`缺失。如果检测器主干还有任何缺失参数或checkpoint存在多余参数，训练会直接停止。

## 6. 如果任务中断

脚本默认不覆盖已有控制台日志。不要删除旧结果后盲目重跑，先确认完成到哪一步：

```bash
cd /root/autodl-tmp/QECR-StereoDETR

ls -lh 'outputs/最终组合_V09/R0_V08O直接复评/控制台.log' 2>/dev/null || true
ls -lh 'outputs/最终组合_V09/V09_V08O加点式三维质量排序/控制台.log' 2>/dev/null || true
```

只补R0：

```bash
bash scripts/运行V09强基线质量组合.sh r0
```

只补R1：

```bash
bash scripts/运行V09强基线质量组合.sh r1
```

两份日志都有后，只重新汇总：

```bash
bash scripts/运行V09强基线质量组合.sh summary
```

如果某一阶段已经留下日志，脚本会拒绝覆盖。需要重跑时应先把旧阶段目录改名归档，再重新执行，而不是把两次运行写入同一日志。

## 7. 预先固定的判定标准

- 通过：R1相对R0的Car Moderate至少提高`+0.10 AP`，Car Hard下降不超过`0.10 AP`。
- 多类别折中保留：Car Moderate有正提升但不足`+0.10 AP`，Car Hard未越线，并且Pedestrian、Cyclist Moderate都提高。
- 否决：其余情况停止，不延长训练、不重新扫描beta、不跑195轮。
- 口径异常：R0未在`49.5345 ± 0.05 AP`内复现时，先检查权重和配置，不解释R1。

`49.7596`只是把V08O历史结果和V06B历史增益机械相加得到的观察目标，不是实验预测值。论文只能使用`V09结果对照.md`中的实测结果。
