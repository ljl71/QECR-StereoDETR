# V22轴向IoU敏感深度损失：实现与预注册

## 1. 基线痛点与证据

StereoDETR使用带不确定性的Laplace型查询深度损失，它优化的是以米为单位的中心深度误差。KITTI三维检测指标实际判断的是三维框IoU是否跨过类别阈值，同样的深度误差对不同尺寸、朝向和距离的目标并不等价。

G7在V09固定检测结果上进行了隔离诊断。轴向IoU损失与真实3D IoU错误的Spearman相关性在Car上达到`0.9122`，比原始绝对深度误差高`0.0742`；只把GT残差的10%用于Oracle修正时，Car Moderate提高`2.5803 AP`。这说明深度仍是有效瓶颈，而且需要更贴近三维重叠的训练目标。

这些数字使用了验证集GT，只是立项证据，不是模型性能。

## 2. 方法

对一个KITTI三维框，沿相机光轴的半占据范围定义为：

```text
r_z = 0.5 * (|l sin(ry)| + |w cos(ry)|)
```

以预测深度和GT深度为中心分别构造一维区间，然后计算一维GIoU损失。GIoU在两个区间不相交时仍有梯度，避免普通IoU的零梯度问题。

实现中预测尺寸、预测朝向、GT尺寸和GT朝向全部从计算图中分离，轴向损失只向预测深度回传梯度。原始尺寸、朝向、中心和Laplace深度损失均保持不变，从而避免把多个回归分量重新纠缠到一个IoU目标中。

## 3. 代码接入位置

- 数学实现：`lib/models/monodetr/axial_depth_iou.py`；
- 目标字段保留：`lib/helpers/qecr_trainer_helper.py`；
- 损失计算与配置路由：`lib/models/monodetr/stereodetr.py`；
- 公平控制：`versions/V22O_原始Laplace深度控制/config.yaml`；
- 候选版本：`versions/V22A_轴向IoU敏感深度损失/config.yaml`；
- 自动实验：`scripts/运行V22轴向IoU深度损失.sh`。

## 4. 开销与公平性

- 新增模型参数：0；
- 新增推理分支：0；
- 理论推理时延增量：0；
- V22O与V22A都只训练`depth_predictor.depth_classifier.*`；
- 两组都从同一个V09最优checkpoint出发，随机种子444、学习率`2e-5`、5轮；
- 唯一变量：V22A启用系数为`0.5`的轴向GIoU辅助损失。

## 5. 止损规则

V22A相对V22O只有在Car Moderate至少`+0.20 AP`，同时Car Easy/Hard和Pedestrian/Cyclist Moderate/Hard均不下降`0.10 AP`以上时，才进入第二随机种子复核。未满足则停止，不运行195轮，也不包装为已验证创新点。

## 6. 直接依据

- [MonoDLE（CVPR 2021论文）](https://openaccess.thecvf.com/content/CVPR2021/papers/Ma_Delving_Into_Localization_Errors_for_Monocular_3D_Object_Detection_CVPR_2021_paper.pdf)：分析三维定位误差，并采用分解后的IoU导向监督，避免多个三维属性在单一IoU目标中互相干扰。
- [MonoDLE官方代码](https://github.com/xinzhuma/monodle)：用于核对其损失分解和实现边界；V22没有复制其网络或主干。
- [MonoDIS（ICCV 2019）](https://arxiv.org/abs/1905.12365)：其核心原则是隔离不同三维参数组的梯度。V22据此将尺寸和朝向从轴向深度辅助损失中分离。

## 7. 2026-08-14正式实验结论

V22O/V22A五轮公平配对和训练后参数审计均已完成。V22A相对V22O的Car 3D AP_R40 Easy/Moderate/Hard变化为`-0.0176/-0.0144/-0.0207`，Pedestrian为`-0.0677/-0.0398/-0.0228`，Cyclist三档均为`0.0000`。两组均无新增/删除模型张量，冻结区变化数为0，说明这是有效的近零负结果，而不是模块未接入或训练范围污染。

最终状态：**V22否决**。不补第二随机种子、不扫描损失系数、不运行195轮，也不作为论文正向创新点。G7仍可作为问题分析和候选筛选证据，但其GT Oracle不能作为模型结果。
