# AutoDL V23分组保真空间视差预聚合实验指南

## 一、上传增量更新包

本地只需上传以下增量包，不需要重新上传完整项目：

`QECR-StereoDETR_GPSD增量更新.zip`

将该文件上传到`/root/autodl-tmp/`，保留服务器原项目和全部`outputs`，然后执行：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
unzip -oq /root/autodl-tmp/QECR-StereoDETR_GPSD增量更新.zip \
  -d /root/autodl-tmp/QECR-StereoDETR
```

增量包只包含5个既有文件的受控修改和21个V23新增文件，不包含`outputs`、checkpoint、数据集或Git目录。原有实验结果不会被删除。脚本启动时会核验项目内V09固定权重SHA256：

`6b673fef2a54e2a5fbc44be731b24914770fffebaf1c20a0cacc43228dd316b4`

若项目内固定V09权重不存在，但权重保存在其他位置，可在运行前指定：

```bash
export V23_SOURCE_CHECKPOINT='/绝对路径/checkpoint_best.pth'
```

## 二、环境初始化

每次新终端先执行：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
```

## 三、第一阶段：训练前检查

```bash
bash scripts/运行V23分组视差预聚合.sh preflight \
  2>&1 | tee V23训练前检查.log
echo "退出码：${PIPESTATUS[0]}"
```

必须看到以下关键信息：

- 六个V23配置检查通过。
- 分组均值恢复原始相关。
- RDSA和GPSD零初始化恒等。
- 新增参数量分别为666和25584。
- V23O/A/B可训练参数均不越过预注册前缀。

退出码不为0时不要训练，把完整`V23训练前检查.log`发回分析。

## 四、第二阶段：G8冻结诊断

推荐在screen中执行：

```bash
screen -dmS v23_g8 bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行V23分组视差预聚合.sh g8
'
```

查看进度：

```bash
tail -n 50 'outputs/V23分组保真预聚合/G8_100样本诊断/运行.log'
```

结束后查看：

```bash
cat 'outputs/V23分组保真预聚合/G8_100样本诊断/诊断报告.md'
```

只有报告写明“通过，允许V23短程配对”才继续。否则停止，不租卡训练V23。

## 五、第三阶段：24样本冒烟

```bash
bash scripts/运行V23分组视差预聚合.sh smoke
```

检查：

```bash
grep -R -E 'exit_code=|Traceback|CUDA out of memory|nan|训练后参数范围审计通过' \
  'outputs/V23分组保真预聚合/冒烟测试' || true
```

## 六、第四阶段：种子444三组配对

V23O、V23A、V23B各训练5轮。三组从同一个V09权重出发，不能只跑候选而省略控制组。

```bash
screen -dmS v23_pair bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行V23分组视差预聚合.sh pair
'
```

查看当前状态：

```bash
bash scripts/运行V23分组视差预聚合.sh status
```

配对结束后：

```bash
cat 'outputs/V23分组保真预聚合/V23配对结果.md'

find 'outputs/V23分组保真预聚合' -maxdepth 3 \
  -type f -name '训练后参数审计.txt' -print -exec cat {} \;
```

若报告为“否决”，立即停止。若为“谨慎”“聚合有效但分组独立性不足”或“精度通过”，再决定是否补第二随机种子。只有“精度通过”能直接支持GPSD的独立贡献。

## 七、第五阶段：种子445复核

```bash
screen -dmS v23_seed445 bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行V23分组视差预聚合.sh seed445_pair
'
```

结束后：

```bash
cat 'outputs/V23分组保真预聚合/V23双随机种子复核报告.md'
```

## 八、第六阶段：正式同步时延

只有双种子报告通过才执行：

```bash
bash scripts/运行V23分组视差预聚合.sh latency
cat 'outputs/V23分组保真预聚合/正式时延测试/时延判定.md'
```

脚本交替运行三组V23O/V23B，统计CUDA同步batch=1模型前向。平均Median增幅上限为3%，参数增幅上限为0.5%。

## 九、第七阶段：QLQC质量重标定

双种子与时延全部通过后运行：

```bash
screen -dmS v23_quality bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行V23分组视差预聚合.sh quality
'
```

结果：

```bash
cat 'outputs/V23分组保真预聚合/V23最终质量重标定结果.md'
```

报告会自动在V23B与V23Q之间选择最终验证权重。V23Q若未带来受保护的正收益，就继续使用V23B。

## 十、需要发回的统计信息

每完成一个阶段，优先发送以下文件内容：

```bash
bash scripts/运行V23分组视差预聚合.sh status

cat 'outputs/V23分组保真预聚合/G8_100样本诊断/诊断报告.md' 2>/dev/null || true
cat 'outputs/V23分组保真预聚合/V23配对结果.md' 2>/dev/null || true
cat 'outputs/V23分组保真预聚合/V23双随机种子复核报告.md' 2>/dev/null || true
cat 'outputs/V23分组保真预聚合/正式时延测试/时延判定.md' 2>/dev/null || true
cat 'outputs/V23分组保真预聚合/V23最终质量重标定结果.md' 2>/dev/null || true

grep -R -E 'Traceback|CUDA out of memory|nan|exit_code=' \
  'outputs/V23分组保真预聚合' || true
```

不要只截取Car Moderate。自动报告同时保留三类目标的Easy、Moderate和Hard，可直接用于判断收益是否来自单类波动。
