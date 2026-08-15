# AutoDL V14 固定相关平滑零训练复评指南

## 结论与目的

G4 正式100样本诊断已经否决 Selective-Stereo 式自适应频率门控：目标边界和内部几乎没有不同的分支选择行为。

但是，同一诊断产生了一个独立且很强的新证据：固定 `s4` 局部平滑在留出集三个类别上都显著改善稀疏视差。V14首先做最便宜的零训练验证：

- V14O：原始 V09，关闭平滑；
- V14S：同一个 V09 checkpoint，开启参数无关的单次 `3×3` 空间平均；
- 两组都不训练，不增加参数；
- 比较正式 KITTI 验证集 3D AP_R40；
- 只有 V14S 达到预注册门槛，才创建5轮共同续训实验。

## 一、上传范围

最稳妥的方式是在 PyCharm 中重新同步整个 `QECR-StereoDETR` 项目。至少需要同步以下文件：

```text
configs/QECR默认配置.yaml
lib/models/monodetr/depth_predictor/depth_predictor_lightstereo.py
versions/V14O_零训练原始相关复评/config.yaml
versions/V14S_零训练s4固定平滑/config.yaml
tools/测试V14固定相关平滑.py
tools/汇总V14固定相关平滑.py
tools/检查QECR消融配置.py
scripts/运行V14固定相关平滑.sh
scripts/运行QECR测试.sh
```

## 二、环境与上传检查

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

ls -lh \
  scripts/运行V14固定相关平滑.sh \
  tools/测试V14固定相关平滑.py \
  tools/汇总V14固定相关平滑.py \
  versions/V14O_零训练原始相关复评/config.yaml \
  versions/V14S_零训练s4固定平滑/config.yaml
```

## 三、训练前测试

```bash
bash scripts/运行QECR测试.sh 2>&1 | tee V14复评前测试.log
V14_TEST_EXIT=${PIPESTATUS[0]}
echo "V14测试退出码：$V14_TEST_EXIT"
```

必须看到：

```text
V14固定相关平滑测试通过
V14结果解析与零训练止损判定自检通过
QECR pre-training checks passed
V14测试退出码：0
```

## 四、运行零训练配对复评

先确认没有训练进程：

```bash
pgrep -af '训练与评估_无蒸馏QECR.py' || echo '当前没有训练/评估进程'
```

然后运行：

```bash
set -o pipefail
bash scripts/运行V14固定相关平滑.sh pair \
  2>&1 | tee V14零训练复评总日志.log

V14_EXIT=${PIPESTATUS[0]}
echo "V14零训练复评退出码：$V14_EXIT"
```

脚本会：

1. 校验 V09 checkpoint；
2. 为 V14O/V14S 各复制一份完全相同的 checkpoint；
3. 比较复制前后的 SHA-256；
4. 分别执行正式 KITTI 验证集评估；
5. 自动生成对照报告。

两组均没有反向传播或参数更新，不需要 `screen`。如果希望防止SSH中断，也可以运行：

```bash
screen -dmS v14_zero_train bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行V14固定相关平滑.sh pair
'
```

## 五、完成后发送结果

```bash
cat 'outputs/固定相关平滑_V14/零训练复评/V14零训练复评结果.md'

echo '===== 退出码与异常 ====='
grep -R -E \
  'exit_code=|Traceback|CUDA out of memory|V14S自动判定' \
  'outputs/固定相关平滑_V14/零训练复评' \
  || true

echo '===== checkpoint一致性 ====='
grep -R -E \
  'source_sha256=|copied_sha256=|training=' \
  'outputs/固定相关平滑_V14/零训练复评' \
  || true
```

把完整输出发回即可。

## 六、预注册止损规则

V14S必须同时满足：

- Car 3D AP_R40 Moderate 相对 V14O 至少 `+0.15 AP`；
- Car Easy、Hard 均不得下降超过 `0.10 AP`；
- Pedestrian/Cyclist 的 Moderate、Hard 均不得下降超过 `0.20 AP`。

若通过，只代表值得建立5轮公平续训配对；还不能直接作为最终创新点。若否决，则说明 G4 的稀疏视差改善被现有 LightStereo 聚合器吸收或不能传递到3D检测，立即停止该路线。
