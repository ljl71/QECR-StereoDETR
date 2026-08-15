#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

V09_DATA='outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/outputs/data'
VAL_FILE='/root/autodl-tmp/datasets/KITTI/object/training/ImageSets/val.txt'
OUTPUT_DIR='outputs/三维框组件误差归因_G5'

if [[ ! -d "$V09_DATA" ]]; then
  echo "找不到V09预测目录：$V09_DATA" >&2
  exit 1
fi
if [[ ! -f "$VAL_FILE" ]]; then
  echo "找不到KITTI验证划分：$VAL_FILE" >&2
  exit 1
fi

prediction_count="$(
  find "$V09_DATA" -maxdepth 1 -type f -name '*.txt' -print \
    | wc -l \
    | tr -d '[:space:]'
)"
validation_count="$(grep -cve '^[[:space:]]*$' "$VAL_FILE")"

echo "V09预测文件数：$prediction_count"
echo "验证样本数：$validation_count"

if [[ "$prediction_count" != "$validation_count" ]]; then
  echo "预测文件数与验证样本数不一致，停止G5，禁止混用其他实验输出。" >&2
  exit 1
fi
if [[ "$validation_count" != '3769' ]]; then
  echo "当前不是标准3769样本验证集，停止G5。" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
python tools/诊断三维框组件误差_G5.py \
  --config 'versions/V09_V08O加点式三维质量排序/config.yaml' \
  --results_dir "$V09_DATA" \
  --split_file "$VAL_FILE" \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/运行.log"

echo "G5完成：$OUTPUT_DIR/诊断报告.md"
