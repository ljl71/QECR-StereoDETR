# AutoDL V20立体几何深度残差执行指南

## 1. 本次上传包含什么

- `lib/models/monodetr/geometry_depth_residual.py`：零初始化、有界的查询级米制深度残差头；
- `lib/models/monodetr/stereodetr.py`：在V09最终深度输出前接入残差头，并增加严格冻结范围；
- `versions/V20O_*`、`V20A_*`、`V20B_*`：零训练控制、查询容量对照和几何正式候选；
- `configs/AutoDL_V20B_立体几何深度残差冒烟.yaml`：24样本一轮冒烟；
- `scripts/运行V20立体几何深度残差.sh`：防覆盖、校验权重、运行和自动汇总；
- `tools/测试V20立体几何深度残差.py`、`检查V20模型结构.py`、`审计V20训练参数.py`：零初始化、参数量、训练前冻结范围、反向传播和训练后逐张量审计；
- G5新增“GT全部三维属性”文本往返审计。

## 2. 上传后的统一环境

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
```

## 3. 先跑全项目检查

```bash
bash scripts/运行QECR测试.sh 2>&1 | tee V20训练前测试.log
echo "测试退出码：${PIPESTATUS[0]}"
```

必须看到：

```text
3 个V20立体几何深度残差配置检查通过
V20深度残差测试通过...
V20结果解析、几何独立贡献和预注册止损判定自检通过
QECR pre-training checks passed
```

## 4. 用一分钟复核G5异常行

旧G5结果中的“GT全部三维属性”接近0 AP不合常理。上传新代码后使用独立输出目录重跑一次：

```bash
python tools/诊断三维框组件误差_G5.py \
  --output_dir outputs/三维框组件误差归因_G5_全属性往返复核 \
  2>&1 | tee outputs/G5全属性往返复核运行.log

echo "G5复核退出码：${PIPESTATUS[0]}"
```

若出现`G5全属性文本往返审计失败`，不要启动V20，把完整报错发回来。若退出码为0，说明轴向深度和单组件结果的序列化路径可信；全属性组合AP仍只作异常记录，不作为V20收益承诺。

## 5. 24样本冒烟

```bash
bash scripts/运行V20立体几何深度残差.sh smoke \
  2>&1 | tee V20冒烟入口.log
```

完成后检查：

```bash
grep -E \
  'loss_depth:|Best Result:|exit_code=|Traceback|CUDA out of memory|nan' \
  'outputs/第三阶段_V20立体几何深度残差/冒烟测试/控制台.log'

cat 'outputs/第三阶段_V20立体几何深度残差/冒烟测试/模型结构检查.txt'
```

只有退出码0、loss有限、可训练参数全部属于`depth_residual_head.*`时才进入配对。

## 6. 后台运行正式五轮配对

`pair`会依次执行V20O零训练复评、V20A五轮、V20B五轮并自动生成报告：

```bash
screen -dmS v20_pair bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行V20立体几何深度残差.sh pair
'

screen -ls
```

每个训练版本结束后，脚本还会自动生成`训练后参数审计.txt`。其中必须出现
`outside_changed_tensor_count=0`和`V20训练后参数范围审计通过`；否则该次
结果不得写入消融表，因为这表示V09冻结区可能发生了变化。

查看进度：

```bash
tail -n 60 'outputs/第三阶段_V20立体几何深度残差/V20A_查询级深度残差/控制台.log'
tail -n 60 'outputs/第三阶段_V20立体几何深度残差/V20B_立体几何深度残差/控制台.log'
```

## 7. 跑完后一次性发回这些信息

```bash
cd /root/autodl-tmp/QECR-StereoDETR

cat 'outputs/第三阶段_V20立体几何深度残差/V20配对结果.md'

echo '===== 异常检查 ====='
grep -R -E \
  'exit_code=|Traceback|CUDA out of memory|nan|V20自动判定' \
  'outputs/第三阶段_V20立体几何深度残差' \
  || true

echo '===== 模型结构 ====='
cat 'outputs/第三阶段_V20立体几何深度残差/V20A_查询级深度残差/模型结构检查.txt'
cat 'outputs/第三阶段_V20立体几何深度残差/V20B_立体几何深度残差/模型结构检查.txt'

echo '===== 训练后冻结参数审计 ====='
cat 'outputs/第三阶段_V20立体几何深度残差/V20A_查询级深度残差/训练后参数审计.txt'
cat 'outputs/第三阶段_V20立体几何深度残差/V20B_立体几何深度残差/训练后参数审计.txt'
```

不要在看到单轮最高值后手工选择某一轮；汇总脚本读取每组最后一次`checkpoint_best`正式复评。五轮未通过门槛时不补到195轮，也不扫描残差上限或学习率。
