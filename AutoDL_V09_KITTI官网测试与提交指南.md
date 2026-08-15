# AutoDL V09 KITTI官网测试与提交指南

## 1. 结论与边界

最终提交模型固定为V09，验证集Car 3D AP_R40 Moderate为`49.8210`，权重SHA256固定为：

```text
6b673fef2a54e2a5fbc44be731b24914770fffebaf1c20a0cacc43228dd316b4
```

本流程不训练、不读取测试标签、不扫描阈值、Top-K或质量指数。V09配置中的`threshold=0.2`、`topk=50`和`score_power=1.5`全部保持验证集阶段已经固定的值。

KITTI官网要求测试服务器只用于最终论文结果，不能用于调参；当前政策还要求方法具有显著创新并面向同行评审论文。申请提交权限时必须如实填写学生身份、单位、目标期刊/会议与方法贡献，申请可能需要人工审核。

## 2. 先申请KITTI提交权限

使用学校/机构邮箱注册并申请submission权限。建议方法描述：

> We study confidence misalignment in efficient stereo DETR-based 3D detection. Our method introduces a lightweight query-level 3D localization-quality head supervised by matched 3D IoU and calibrates the final ranking score using class, depth and learned localization quality. The detector adds only 0.36% parameters and is being prepared for submission to a peer-reviewed journal.

不要写“只是复现”“简单加了一个头”，也不要夸大为多个互不相关的模块；按照真实的完整问题—方法—实验链描述。

## 3. 检查数据盘空间并解压官方test数据

```bash
df -h /root/autodl-tmp

RAW=/autodl-pub/data/KITTI_Object/raw
ROOT=/root/autodl-tmp/datasets/KITTI/object

mkdir -p "$ROOT"

for file in \
  data_object_calib.zip \
  data_object_image_2.zip \
  data_object_image_3.zip
do
  echo "正在解压test部分：$file"
  unzip -q -o "$RAW/$file" 'testing/*' -d "$ROOT"
done
```

不需要测试标签，也不需要Velodyne点云。预计还需要约12GB左右数据空间；以实际`df -h`为准。

## 4. 上传新代码后运行最终推理

确认下列三个新增/修改文件已经同步：

```text
lib/helpers/tester_helper.py
versions/V09_KITTI官网测试提交/config.yaml
scripts/运行V09_KITTI官网测试提交.sh
```

运行：

```bash
cd /root/autodl-tmp/QECR-StereoDETR
source /root/miniconda3/bin/activate stereodetr-open

export CUDA_HOME=/usr/local/cuda
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

bash scripts/运行V09_KITTI官网测试提交.sh
```

脚本将自动完成：

1. 校验V09权重SHA256；
2. 校验7518对`image_2/image_3/calib`；
3. 对KITTI test执行一次推理；
4. 跳过不存在真值的本地评估；
5. 校验7518个结果文件和每条记录的16字段格式；
6. 生成txt直接位于ZIP根目录的提交包。

最终文件：

```text
outputs/KITTI官网测试_V09/V09_KITTI_test_submission.zip
outputs/KITTI官网测试_V09/V09_KITTI_test_submission.zip.sha256
```

## 5. 上传前最后检查

```bash
cd /root/autodl-tmp/QECR-StereoDETR

unzip -Z1 \
  outputs/KITTI官网测试_V09/V09_KITTI_test_submission.zip \
  | wc -l

unzip -Z1 \
  outputs/KITTI官网测试_V09/V09_KITTI_test_submission.zip \
  | head

cat \
  outputs/KITTI官网测试_V09/V09_KITTI_test_submission.zip.sha256
```

应看到文件数`7518`，文件名从`000000.txt`开始，且没有`data/`目录前缀。然后将该ZIP上传到KITTI object/3D object detection提交页面，并标记输入信息为Stereo，不使用Laser、Flow或Multiview。

官网结果只作为最终测试结果使用。不要根据返回结果重新选择beta、阈值、checkpoint或再次上传另一个版本。
