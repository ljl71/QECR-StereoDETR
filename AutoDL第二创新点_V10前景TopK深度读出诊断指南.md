# AutoDL第二创新点候选：V10前景Top-K深度读出诊断指南

## 1. 这次为什么不训练

StereoDETR当前把80个前景深度bin和1个背景bin一起Softmax，再对完整分布求期望。当前候选针对多峰前景分布可能产生“两个峰之间的虚假平均深度”这一明确痛点，借鉴CoEx在soft-argmin前进行Top-K选择的机制。

本轮只使用已经完成的V09最佳checkpoint，在相同Chen验证集上替换参数无关的深度读出。三组都不训练，预计每组约3分钟。只有冻结权重诊断通过，才继续讨论正式结构与消融；未通过就立即停止，避免租卡浪费。

## 2. 三个严格配对版本

| 版本 | 深度读出 | 作用 |
|---|---|---|
| V10O | 原始81类全分布期望 | 检查新代码关闭时是否精确恢复V09的49.8210 |
| V10A | 80个前景bin中Top-K=2 | 预注册主候选，来源对应CoEx默认风格 |
| V10B | 80个前景bin中Top-K=4 | 只检查机制敏感性，不作为事后调参 |

Top-K不会删除背景判断。实现先保留原81类Softmax得到的背景概率，再把剩余前景概率质量重新分配给Top-K前景bin：

```text
d = (1 - p_bg) * Σ softmax(topk_foreground_logits) * depth_bin + p_bg * depth_max
```

因此，当K覆盖全部80个前景bin时，该公式数学上恢复原始全分布期望；背景像素不会被强行解释为近处目标。

## 3. 上传注意事项

将本地QECR-StereoDETR代码同步到：

```text
/root/autodl-tmp/QECR-StereoDETR
```

不要启用“删除服务器中本地不存在的文件”。下面这个服务器权重必须保留：

```text
/root/autodl-tmp/QECR-StereoDETR/outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth
```

其已记录SHA256为：

```text
6b673fef2a54e2a5fbc44be731b24914770fffebaf1c20a0cacc43228dd316b4
```

## 4. 训练前测试

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
V10前景Top-K深度读出测试通过
V10结果解析、主候选优先级与预注册止损判定自检通过
QECR pre-training checks passed
测试退出码：0
```

## 5. 一条命令完成三个冻结权重评估

建议使用screen：

```bash
screen -S v10_topk_depth
```

进入screen后执行：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

bash scripts/运行V10前景TopK深度诊断.sh all
```

按`Ctrl+A`再按`D`可退出screen，不中断程序。

## 6. 输出目录

```text
outputs/第二创新点_V10前景TopK深度读出诊断/
├── V10O_全分布控制/
├── V10A_前景TopK2/
├── V10B_前景TopK4/
└── V10诊断结果.md
```

三个输出目录各自保存配置快照、运行信息、控制台日志、复制的V09权重和3769个验证集检测结果。运行信息同时记录源权重与复制权重的SHA256。

## 7. 预先固定的通过与止损规则

V10A的K=2是主候选，必须同时满足：

- Car 3D AP_R40 Moderate相对V10O提高至少`+0.15 AP`；
- Car Hard下降不超过`0.10 AP`；
- Pedestrian Moderate下降不超过`0.10 AP`；
- Cyclist Moderate下降不超过`0.10 AP`。

判定规则：

- K=2通过：进入正式模块、时延和必要消融设计；
- K=2不通过但K=4通过：只记为“待独立复核”，不能直接写成创新点；
- 两者均不通过：立即否决，不扫描更多K、温度或门控阈值，也不重新训练。
- V10O未恢复`49.8210±0.05`：属于口径异常，先检查代码、权重和配置，不解释Top-K结果。

## 8. 完成后需要回传的信息

```bash
cd /root/autodl-tmp/QECR-StereoDETR

cat 'outputs/第二创新点_V10前景TopK深度读出诊断/V10诊断结果.md'

cat 'outputs/第二创新点_V10前景TopK深度读出诊断/V10O_全分布控制/运行信息.txt'
cat 'outputs/第二创新点_V10前景TopK深度读出诊断/V10A_前景TopK2/运行信息.txt'
cat 'outputs/第二创新点_V10前景TopK深度读出诊断/V10B_前景TopK4/运行信息.txt'

grep -R -E 'Traceback|CUDA out of memory|nan|inf|exit_code=' \
  'outputs/第二创新点_V10前景TopK深度读出诊断' \
  | tail -n 30
```

