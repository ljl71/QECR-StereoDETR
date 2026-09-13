#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/运行Car专项质量校准.sh preflight
#   bash scripts/运行Car专项质量校准.sh smoke
#   bash scripts/运行Car专项质量校准.sh train
#   bash scripts/运行Car专项质量校准.sh infer
#   bash scripts/运行Car专项质量校准.sh all
#   bash scripts/运行Car专项质量校准.sh status

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|train|infer|all|status) ;;
  *)
    echo "错误：未知模式 ${MODE}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SOURCE_CHECKPOINT="outputs/KITTI_trainval官方批量16复核/T2_QLQC/qlqc_trainval_global_batch16/checkpoint_final.pth"
TRAIN_CONFIG="configs/Car专项质量校准/T3_trainval_Car残差质量.yaml"
SMOKE_CONFIG="configs/Car专项质量校准/Car残差质量冒烟.yaml"
EVAL_CONFIG="configs/Car专项质量校准/Car残差质量_KITTI测试.yaml"
TRAIN_DIR="outputs/Car专项质量校准/T3_Car残差质量/qlqc_car_residual_calibration"
SMOKE_DIR="outputs/Car专项质量校准/冒烟测试/qlqc_car_residual_smoke"
EVAL_DIR="outputs/Car专项质量校准/官网测试/qlqc_car_residual_kitti_test"
TRAINED_CHECKPOINT="${TRAIN_DIR}/checkpoint_final.pth"
SMOKE_CHECKPOINT="${SMOKE_DIR}/checkpoint_final.pth"
EVAL_CHECKPOINT="${EVAL_DIR}/checkpoint_best.pth"
RESULT_DIR="${EVAL_DIR}/outputs/data"
ZIP_FILE="outputs/Car专项质量校准/QLQC_Car_residual_KITTI_test_submission.zip"
SMOKE_LIST="/root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train_car_quality_smoke.txt"

die() {
  echo "错误：$*" >&2
  exit 3
}

require_environment() {
  if [[ "${CONDA_DEFAULT_ENV:-}" != "stereodetr-open" ]]; then
    die "请先执行 source /root/miniconda3/bin/activate stereodetr-open"
  fi
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export NUMBA_CUDA_USE_NVIDIA_BINDING="${NUMBA_CUDA_USE_NVIDIA_BINDING:-1}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export PYTHONUNBUFFERED=1
}

validate_checkpoint() {
  local checkpoint="$1"
  local expected_epoch="$2"
  python - "${checkpoint}" "${expected_epoch}" <<'PY'
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit("缺少checkpoint：{}".format(path))
checkpoint = torch.load(path, map_location="cpu")
if int(checkpoint.get("epoch", -1)) != expected:
    raise SystemExit(
        "{}的epoch={}，期望{}".format(
            path, checkpoint.get("epoch"), expected
        )
    )
state = checkpoint.get("model_state")
if not isinstance(state, dict) or not state:
    raise SystemExit("{}不含有效model_state".format(path))
print("checkpoint通过：{}，epoch={}，tensor={}".format(path, expected, len(state)))
PY
}

count_results() {
  if [[ -d "${RESULT_DIR}" ]]; then
    find "${RESULT_DIR}" -maxdepth 1 -type f -name '*.txt' | wc -l
  else
    echo 0
  fi
}

validate_dataset() {
  local trainval_list="/root/autodl-tmp/datasets/KITTI/object/training/ImageSets/trainval.txt"
  local test_list="/root/autodl-tmp/datasets/KITTI/object/testing/ImageSets/test.txt"
  [[ -f "${trainval_list}" ]] || die "缺少${trainval_list}"
  [[ -f "${test_list}" ]] || die "缺少${test_list}"
  local trainval_count test_count
  trainval_count="$(wc -l < "${trainval_list}" | tr -d '[:space:]')"
  test_count="$(wc -l < "${test_list}" | tr -d '[:space:]')"
  [[ "${trainval_count}" == "7481" ]] \
    || die "trainval列表为${trainval_count}/7481"
  [[ "${test_count}" == "7518" ]] \
    || die "test列表为${test_count}/7518"
  local directory file_count
  for directory in calib image_2 image_3; do
    file_count="$(
      find "/root/autodl-tmp/datasets/KITTI/object/testing/${directory}" \
        -maxdepth 1 -type f | wc -l | tr -d '[:space:]'
    )"
    [[ "${file_count}" == "7518" ]] \
      || die "KITTI test ${directory}为${file_count}/7518"
  done
  echo "KITTI数据通过：trainval 7481张，test 7518张双目图像与标定"
}

run_preflight() {
  python tools/检查Car专项质量校准.py
  python tools/测试3D质量排序.py
  python -m py_compile \
    lib/models/monodetr/quality_ranking.py \
    lib/models/monodetr/stereodetr.py \
    tools/检查Car专项质量校准.py \
    tools/审计Car专项质量校准权重.py
  validate_checkpoint "${SOURCE_CHECKPOINT}" 3
  validate_dataset
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA不可用")
properties = torch.cuda.get_device_properties(0)
print("GPU: {}，显存{:.2f} GiB".format(
    properties.name, properties.total_memory / 1024 ** 3
))
PY
  echo "Car专项质量校准预检通过"
}

run_smoke() {
  mkdir -p "$(dirname "${SMOKE_LIST}")" "$(dirname "${SMOKE_DIR}")"
  head -n 32 \
    /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/trainval.txt \
    > "${SMOKE_LIST}"
  set +e
  python tools/训练与评估_无蒸馏QECR.py \
    --config "${SMOKE_CONFIG}" \
    --batch_size 16 \
    --gpu_ids 0 \
    --amp \
    2>&1 | tee "$(dirname "${SMOKE_DIR}")/控制台.log"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  [[ "${exit_code}" -eq 0 ]] || die "Car专项冒烟失败，退出码=${exit_code}"
  validate_checkpoint "${SMOKE_CHECKPOINT}" 1
  python tools/审计Car专项质量校准权重.py \
    --source "${SOURCE_CHECKPOINT}" \
    --trained "${SMOKE_CHECKPOINT}" \
    --output "$(dirname "${SMOKE_DIR}")/训练后参数审计.txt"
  echo "Car专项质量校准冒烟通过"
}

run_train() {
  if validate_checkpoint "${TRAINED_CHECKPOINT}" 3 >/dev/null 2>&1; then
    echo "Car专项质量头已经完成3轮训练，跳过"
  else
    mkdir -p "$(dirname "${TRAIN_DIR}")"
    set +e
    python tools/训练与评估_无蒸馏QECR.py \
      --config "${TRAIN_CONFIG}" \
      --batch_size 16 \
      --gpu_ids 0 \
      --amp \
      2>&1 | tee -a "$(dirname "${TRAIN_DIR}")/控制台.log"
    local exit_code="${PIPESTATUS[0]}"
    set -e
    [[ "${exit_code}" -eq 0 ]] || die "Car专项训练失败，退出码=${exit_code}"
  fi
  validate_checkpoint "${TRAINED_CHECKPOINT}" 3
  python tools/审计Car专项质量校准权重.py \
    --source "${SOURCE_CHECKPOINT}" \
    --trained "${TRAINED_CHECKPOINT}" \
    --output "$(dirname "${TRAIN_DIR}")/训练后参数审计.txt"
}

run_infer() {
  validate_checkpoint "${TRAINED_CHECKPOINT}" 3
  local count
  count="$(count_results | tr -d '[:space:]')"
  if [[ "${count}" != "0" && "${count}" != "7518" ]]; then
    die "官网测试目录已有${count}/7518个txt，请先人工备份并移走不完整目录"
  fi
  if [[ "${count}" == "0" ]]; then
    mkdir -p "${EVAL_DIR}" "$(dirname "${EVAL_DIR}")"
    if [[ -f "${EVAL_CHECKPOINT}" ]]; then
      local source_hash target_hash
      source_hash="$(sha256sum "${TRAINED_CHECKPOINT}" | awk '{print $1}')"
      target_hash="$(sha256sum "${EVAL_CHECKPOINT}" | awk '{print $1}')"
      [[ "${source_hash}" == "${target_hash}" ]] \
        || die "官网测试目录中的checkpoint与Car专项最终权重不一致"
    else
      cp "${TRAINED_CHECKPOINT}" "${EVAL_CHECKPOINT}"
    fi
    set +e
    python tools/训练与评估_无蒸馏QECR.py \
      --evaluate_only \
      --config "${EVAL_CONFIG}" \
      --batch_size 12 \
      --gpu_ids 0 \
      --no_amp \
      2>&1 | tee "$(dirname "${EVAL_DIR}")/控制台.log"
    local exit_code="${PIPESTATUS[0]}"
    set -e
    [[ "${exit_code}" -eq 0 ]] || die "Car专项KITTI test推理失败"
  fi
  count="$(count_results | tr -d '[:space:]')"
  [[ "${count}" == "7518" ]] || die "结果文件数为${count}/7518"
  python tools/检查并打包KITTI提交.py \
    --result_dir "${RESULT_DIR}" \
    --output_zip "${ZIP_FILE}"
  echo "最终提交包：${PROJECT_ROOT}/${ZIP_FILE}"
}

show_status() {
  echo "===== Car专项质量校准状态 ====="
  validate_checkpoint "${SOURCE_CHECKPOINT}" 3 >/dev/null 2>&1 \
    && echo "完成  全量QLQC起点" \
    || echo "缺少  全量QLQC起点"
  validate_checkpoint "${SMOKE_CHECKPOINT}" 1 >/dev/null 2>&1 \
    && echo "完成  冒烟测试" \
    || echo "未完成  冒烟测试"
  validate_checkpoint "${TRAINED_CHECKPOINT}" 3 >/dev/null 2>&1 \
    && echo "完成  Car残差质量头3轮" \
    || echo "未完成  Car残差质量头3轮"
  echo "KITTI结果文件数：$(count_results | tr -d '[:space:]') / 7518"
  [[ -f "${ZIP_FILE}" ]] && ls -lh "${ZIP_FILE}" "${ZIP_FILE}.sha256" || true
}

require_environment
if [[ "${MODE}" == "status" ]]; then
  show_status
  exit 0
fi

run_preflight
case "${MODE}" in
  preflight) ;;
  smoke) run_smoke ;;
  train) run_train ;;
  infer) run_infer ;;
  all)
    run_smoke
    run_train
    run_infer
    ;;
esac
show_status
