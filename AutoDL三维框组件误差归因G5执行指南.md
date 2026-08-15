# AutoDL三维框组件误差归因G5执行指南

## 这一步做什么

G5只读取V09已经保存的KITTI验证集预测，对三维框的深度、投影中心、尺寸和朝向做逐组件真值替换诊断。它不训练、不加载V09模型、不改checkpoint，也不覆盖V09原始预测。

输出中的所有`Oracle`目录都用了验证集真值，只能用于判断下一创新方向，禁止把这些数值写成模型精度或提交KITTI。

## 一、上传后检查文件

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

ls -lh \
  'tools/诊断三维框组件误差_G5.py' \
  'scripts/运行G5三维框组件误差归因.sh' \
  '下一阶段改进方向全面分析与G5误差归因预注册.md' \
  'AutoDL三维框组件误差归因G5执行指南.md'
```

## 二、设置既有环境

```bash
export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
```

不需要创建新conda环境，也不需要重新编译算子。

## 三、先运行训练前测试

```bash
set -o pipefail

bash scripts/运行QECR测试.sh \
  2>&1 | tee G5诊断前测试.log

TEST_EXIT=${PIPESTATUS[0]}
echo "测试退出码：$TEST_EXIT"
```

应看到：

```text
G5自检通过：2D匹配、角度环绕、投影中心、沿射线深度替换与KITTI文本读回正常
QECR pre-training checks passed
测试退出码：0
```

## 四、运行正式G5诊断

```bash
bash scripts/运行G5三维框组件误差归因.sh
```

脚本会先检查V09预测文件数和验证样本数是否都为`3769`；任何一个不满足都会退出，避免混用其他实验输出。命令正常返回提示符即表示退出码为0，也可以紧接着执行`echo "G5退出码：$?"`复核。

脚本会对原始V09和六个组件Oracle调用项目内KITTI评估。它会占用GPU做旋转IoU评估，但不会执行网络前向和训练。预计通常为数分钟到十几分钟，具体取决于共享存储和Numba首次编译。

## 五、完成后把这些信息发给我

```bash
cat 'outputs/三维框组件误差归因_G5/诊断报告.md'

echo '===== 异常检查 ====='
grep -E \
  'Traceback|CUDA out of memory|自动结论:|下一候选:' \
  'outputs/三维框组件误差归因_G5/运行.log' \
  || true

echo '===== 产物 ====='
find 'outputs/三维框组件误差归因_G5' \
  -maxdepth 2 \
  -type f \
  -printf '%p  %k KB\n' \
  | sort
```

请把上述完整输出发回。最重要的是：

- V09原始预测三类3D AP_R40是否恢复历史数值；
- 每个组件Oracle相对V09的Car/Pedestrian/Cyclist Moderate变化；
- 自动结论和下一候选；
- 是否有Traceback、CUDA OOM或路径异常。

## 六、不要做的操作

- 不要运行195轮训练；
- 不要自行创建V15配置；
- 不要把`Oracle_仅诊断禁止提交`中的文本当作模型预测；
- 不要把Oracle数值记录为论文精度；
- 不要覆盖或删除V09 checkpoint和原始`outputs/data`。

只有G5通过预注册门槛后，才会在本地实现对应的小规模配对版本，并给出新的上传文件清单与AutoDL命令。
