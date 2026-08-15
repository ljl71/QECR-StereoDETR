# AutoDL第二创新点：V12轻量分组相关门控实验指南

## 1. 当前结论

V12-G0在100个固定样本、60983个稀疏参照点上完成正式诊断，自动结论为：
**通过进入轻量门控短程配对**。

- 最佳组合：s4、16组；
- 目标点：24022；Car点：20372；
- 目标区域Oracle ±1 bin命中率提升30.03个百分点，MAE相对下降73.74%；
- Car Oracle ±1 bin命中率提升30.65个百分点，MAE相对下降73.01%；
- 分组最大值比原均值更差，因此禁止使用固定max聚合；
- Oracle使用激光真值选组，只表示信息上限，不能作为模型结果。

正式V12A采用零初始化空间门控。初始softmax为均匀的1/16，因此首次前向与原始全通道平均等价。V12O和V12A都只续训同一个`cost_agg`分支；V12A额外训练416个门控参数。

## 2. 重新上传后检查

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

bash scripts/运行QECR测试.sh
echo "测试退出码：$?"
```

必须看到：

```text
2 个V12分组相关门控配置检查通过
V12轻量分组相关测试通过
V12结果解析与预注册止损判定自检通过
QECR pre-training checks passed
测试退出码：0
```

## 3. 先跑24样本冒烟

```bash
bash scripts/运行V12轻量分组相关.sh smoke
echo "冒烟退出码：$?"
```

冒烟只验证checkpoint兼容、前向反向、保存、显存和损失有限性，不比较AP。

## 4. 运行5轮同起点配对

```bash
pgrep -af '训练与评估_无蒸馏QECR.py' || echo "当前没有训练进程"

screen -dmS v12_pair bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export LANG=C.UTF-8 &&
export LC_ALL=C.UTF-8 &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行V12轻量分组相关.sh pair
'
```

查看状态：

```bash
screen -ls
pgrep -af '训练与评估_无蒸馏QECR.py' || echo "V12配对已经结束"
tail -n 40 \
  'outputs/第二创新点_V12分组相关门控/V12A_s4十六组轻量门控/控制台.log'
```

如果V12O已经存在但V12A尚未运行，可单独执行：

```bash
bash scripts/运行V12轻量分组相关.sh candidate
```

## 5. 输出配对结论

```bash
bash scripts/运行V12轻量分组相关.sh summary
cat 'outputs/第二创新点_V12分组相关门控/V12配对结果.md'
```

预注册判定：

- 精度通过：Car 3D AP_R40 Moderate至少提高0.15，同时Easy、Hard及小类别不触发退化保护；
- 谨慎：Car Moderate提高0.05–0.15，先补第二随机种子；
- 否决：不满足以上条件，停止V12完整训练。

## 6. 只有“精度通过”才测时延

```bash
bash scripts/运行V12轻量分组相关.sh latency
cat 'outputs/第二创新点_V12分组相关门控/正式时延测试/汇总.txt'
```

脚本采用O→A、A→O、O→A三组交替顺序，每次20轮预热、100轮同步计时。候选的Median时延增幅必须不超过3%。

## 7. 需要发回的内容

```bash
cat 'outputs/第二创新点_V12分组相关门控/V12配对结果.md'
cat 'outputs/第二创新点_V12分组相关门控/正式时延测试/汇总.txt' 2>/dev/null || true
```

不要删除V09、V12O或V12A checkpoint。短程通过不等于论文结论，还需要第二随机种子和时延复核。

## 8. 等价优化复测通过后的种子445复核

重新上传种子445配置、双种子汇总器和更新后的运行脚本，先执行完整训练前测试：

```bash
bash scripts/运行QECR测试.sh
```

测试必须显示`4 个V12分组相关门控配置检查通过`、`V12双随机种子复核与收紧保护门槛自检通过`以及最终的`QECR pre-training checks passed`。然后后台运行：

```bash
screen -dmS v12_seed445 bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行V12轻量分组相关.sh seed445_pair
'
```

完成后查看：

```bash
cat 'outputs/第二创新点_V12分组相关门控/V12双随机种子复核报告.md'
```

训练前固定的复核条件为：种子445 Car Moderate至少`+0.10`，两种子配对增益平均至少`+0.15`；平均Car Easy/Hard均不低于`-0.10`；平均Pedestrian/Cyclist Moderate不低于`-0.30`、Hard不低于`-0.75`。报告还会单独判断两个V12A的Car Moderate平均是否达到V09的`49.8210`。前一项回答模块是否可复现，后一项回答最终结果是否真的变强，两者不得混写。
