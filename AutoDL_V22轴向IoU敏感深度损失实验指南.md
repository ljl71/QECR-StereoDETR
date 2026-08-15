# AutoDL V22轴向IoU敏感深度损失实验指南

## 结论先行

G7已经证明：在固定候选与其他三维属性时，小幅修正轴向深度可以显著改变KITTI三维IoU阈值通过情况，而且轴向1D IoU损失与真实3D IoU错误的相关性高于原始绝对深度误差。因此V22只新增一个**训练期、参数为零、推理开销为零**的轴向GIoU辅助损失。

V22不直接跑195轮。先进行24样本冒烟，再做从同一个V09最优权重出发的5轮公平配对：

- `V22O`：原始Laplace查询深度损失，续训原有`depth_classifier`；
- `V22A`：训练范围、起点、随机种子和学习率完全相同，只额外加入轴向1D GIoU损失。

## 一、上传后执行完整预检

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

只有看到`QECR pre-training checks passed`且退出码为0才继续。

## 二、先跑24样本冒烟

```bash
bash scripts/运行V22轴向IoU深度损失.sh smoke
```

检查：

```bash
grep -E \
  'loss_depth_axial_iou|Best Result:|Traceback|CUDA out of memory|nan|exit_code=' \
  'outputs/第三阶段_V22轴向IoU深度损失/冒烟测试/控制台.log'

cat \
  'outputs/第三阶段_V22轴向IoU深度损失/冒烟测试/训练后参数审计.txt'
```

必须同时满足：

1. 日志出现有限的`loss_depth_axial_iou`；
2. 无Traceback、OOM和NaN；
3. `exit_code=0`；
4. 参数审计显示新增/删除参数均为0，冻结区变化为0。

## 三、启动5轮公平配对

冒烟通过后：

```bash
screen -dmS v22_pair bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行V22轴向IoU深度损失.sh pair
'
```

确认启动：

```bash
screen -ls
pgrep -af '训练与评估_无蒸馏QECR.py' || true
```

查看当前阶段日志：

```bash
find 'outputs/第三阶段_V22轴向IoU深度损失' \
  -maxdepth 2 -type f -name '控制台.log' -print
```

把上一步找到的具体路径传给`tail -f`，不要把通配符放在单引号内。

## 四、结束后收集结果

```bash
cat 'outputs/第三阶段_V22轴向IoU深度损失/V22配对结果.md'

cat \
  'outputs/第三阶段_V22轴向IoU深度损失/V22O_原始Laplace深度控制/训练后参数审计.txt'

cat \
  'outputs/第三阶段_V22轴向IoU深度损失/V22A_轴向IoU敏感深度损失/训练后参数审计.txt'

grep -R -E \
  'Traceback|CUDA out of memory|nan|exit_code=|V22自动判定' \
  'outputs/第三阶段_V22轴向IoU深度损失' \
  || true
```

将这四段输出发回即可。

## 五、预注册决策线

- `精度通过`：V22A相对V22O的Car 3D AP_R40 Moderate至少`+0.20`，Car Easy/Hard以及Pedestrian/Cyclist Moderate/Hard均不低于`-0.10`；下一步做第二随机种子复核。
- `谨慎`：Car Moderate达到`+0.10`但不足`+0.20`，且保护线通过；只补第二随机种子，不跑195轮。
- `否决`：未满足上述条件；停止V22。

G7中的GT替换和Oracle只能解释机制，不能作为论文中的模型精度。真正可写入实验表的是本次V22O/V22A正式KITTI验证集配对结果。
