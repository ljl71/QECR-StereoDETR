# V09 使用 KITTI trainval 重训并生成唯一官网提交包

## 1. 这次实验的目的

验证集实验已经完成模型选择和消融分析。最终官网复核不再比较消融版本，也不再扫描超参数，只执行一条预先固定的训练链：

1. 使用 Chen 划分的 `3712/3769` 完成方法选择、参数选择与消融实验。
2. 参数冻结后，将两部分合并为 `trainval`，使用全部 `7481` 张有标签双目图训练最终模型。
3. 使用固定轮次训练 `V09T0 → V09T1 → V09T2`。
4. 对 `7518` 张无标签 KITTI test 图像推理一次。
5. 只生成并提交一个 QLQC/V09 ZIP。

KITTI test 图像没有参与训练，也不能根据官网返回结果继续调参。这里的“全部数据集训练”仅指全部 `7481` 张有标签训练图，绝不包含 `testing` 目录。

## 2. 固定训练协议

| 阶段 | 训练数据 | 固定轮次 | 作用 |
|---|---:|---:|---|
| V09T0 | KITTI trainval 7481 | 195 | 重新训练 StereoDETR 基础检测器 |
| V09T1 | KITTI trainval 7481 | 1 | 复现验证阶段预先选定的强控制续训配方 |
| V09T2 | KITTI trainval 7481 | 3 | 冻结检测器，只训练 QLQC 查询级三维质量头 |
| 官网推理 | KITTI test 7518 | 0 | 读取 V09T2 固定轮次权重，仅推理和打包 |

`195 + 1 + 3` 和质量分数指数 `1.5` 都来自此前验证集实验，在 KITTI test 推理前已经固定。训练过程不创建验证集最佳权重，而是保存轮次含义明确的 `checkpoint_final.pth`。

## 3. 上传注意事项

将本地最新源码同步到：

```text
/root/autodl-tmp/QECR-StereoDETR
```

不要让 PyCharm 删除服务器上的 `outputs`、`datasets` 和已编译 CUDA 算子。建议只上传新增或修改的源码文件。

## 4. 预检

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

bash scripts/运行V09_trainval官网最终提交.sh preflight
```

预检会确认：

- `trainval.txt` 无重复覆盖 `000000` 到 `007480`
- 7481 份训练标定、左右图、标签完整
- `test.txt` 按顺序覆盖 `000000` 到 `007517`
- 7518 份测试标定和左右图完整
- 三个训练阶段的继承关系、轮数和质量头参数没有漂移

## 5. 在 screen 中运行完整流程

RTX 4090 上整个 trainval 重训预计约需 `28–36` 小时，实际时间受数据读取和实例性能影响。必须使用 `screen`：

```bash
screen -dmS v09_trainval_final bash -lc '
cd /root/autodl-tmp/QECR-StereoDETR &&
source /root/miniconda3/bin/activate stereodetr-open &&
export CUDA_HOME=/usr/local/cuda &&
export NUMBA_CUDA_USE_NVIDIA_BINDING=1 &&
export CUDA_VISIBLE_DEVICES=0 &&
export OMP_NUM_THREADS=8 &&
bash scripts/运行V09_trainval官网最终提交.sh all
'
```

查看会话：

```bash
screen -ls
screen -r v09_trainval_final
```

从 screen 暂时退出但不中断训练：按 `Ctrl+A`，松开后按 `D`。

## 6. 查看状态与日志

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

bash scripts/运行V09_trainval官网最终提交.sh status

tail -n 60 \
  'outputs/KITTI_trainval官网复核/V09T0_trainval基础模型/控制台.log'
```

三个阶段分别记录独立日志。若 AutoDL 关机或 SSH 中断，重新执行同一条 `all` 命令即可。完成阶段会自动跳过，未完成阶段会从各自的 `checkpoint.pth` 续跑。

## 7. 最终产物

正常完成后只提交这个文件：

```text
outputs/KITTI_trainval官网复核/V09_trainval_QLQC_KITTI_test_submission.zip
```

同时会生成：

```text
outputs/KITTI_trainval官网复核/V09_trainval_QLQC_KITTI_test_submission.zip.sha256
```

打包工具会逐项检查：

- ZIP 内恰好有 7518 个 TXT
- 文件编号连续且全部位于 ZIP 根目录
- 每条检测为 KITTI 16 字段格式
- 类别仅为 Car、Pedestrian、Cyclist
- 数值中不存在 NaN 或 Inf
- ZIP 可完整解压

不需要提交 V09T0、V09T1 或任何消融版本。官网结果用于与 StereoDETR 公开测试结果作一次最终对照，论文中的消融表仍采用原有 Chen 验证集结果。
