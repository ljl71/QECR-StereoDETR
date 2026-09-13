#!/usr/bin/env bash
set -euo pipefail

# 同环境公平基线：trainval固定195轮纯StereoDETR，再对KITTI test推理一次。
# 不运行V09T1强控制，不运行V09T2 QLQC，不根据隐藏test结果选优或调参。

MODE="${1:-all}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_CONFIG="configs/V09_trainval官网复核/V09T0_trainval基础模型.yaml"
EVAL_CONFIG="configs/V09_trainval官网复核/V09E0_StereoDETR基线官网测试.yaml"

RUN_ROOT="outputs/KITTI_trainval官网复核"
TRAIN_DIR="${RUN_ROOT}/V09T0_trainval基础模型/qecr_v09t0_trainval_stereodetr"
SOURCE_CHECKPOINT="${TRAIN_DIR}/checkpoint_final.pth"
EVAL_ROOT="${RUN_ROOT}/官网测试/StereoDETR同环境基线"
EVAL_DIR="${EVAL_ROOT}/qecr_v09e0_stereodetr_reimpl_kitti_test"
TARGET_CHECKPOINT="${EVAL_DIR}/checkpoint_best.pth"
RESULT_DIR="${EVAL_DIR}/outputs/data"
SUBMISSION_ZIP="${RUN_ROOT}/StereoDETR_Reimpl_trainval_KITTI_test_submission.zip"
TRAIN_LOG="${RUN_ROOT}/V09T0_trainval基础模型/控制台.log"
TRAIN_INFO="${RUN_ROOT}/V09T0_trainval基础模型/运行信息.txt"
EVAL_LOG="${EVAL_ROOT}/控制台.log"
EVAL_INFO="${EVAL_ROOT}/运行信息.txt"
PROTOCOL_FILE="${EVAL_ROOT}/固定基线协议与环境.txt"

case "${MODE}" in
  all|preflight|train|infer|status)
    ;;
  *)
    echo "用法：bash scripts/运行StereoDETR同环境基线官网提交.sh [all|preflight|train|infer|status]" >&2
    exit 2
    ;;
esac

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

checkpoint_is_valid() {
  local checkpoint="$1"
  local expected_epoch="$2"
  python - "${checkpoint}" "${expected_epoch}" <<'PY'
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1])
expected_epoch = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
checkpoint = torch.load(path, map_location="cpu")
if int(checkpoint.get("epoch", -1)) != expected_epoch:
    raise SystemExit(
        f"{path}轮次异常：{checkpoint.get('epoch')}，期望{expected_epoch}"
    )
state = checkpoint.get("model_state")
if not isinstance(state, dict) or not state:
    raise SystemExit(f"{path}不含有效model_state")
print(f"固定轮次checkpoint通过：{path}，epoch={expected_epoch}，tensor={len(state)}")
PY
}

run_preflight() {
  python tools/检查StereoDETR同环境基线配置.py
  # 复用已经审计过的数据完整性、固定轮次和checkpoint兼容测试。
  bash scripts/运行V09_trainval官网最终提交.sh preflight
  echo "StereoDETR同环境基线训练与官网推理预检通过"
}

write_protocol_record() {
  mkdir -p "${EVAL_ROOT}"
  {
    echo "protocol=stereodetr_reimpl_same_environment_trainval_fixed_epoch"
    echo "method=pure_stereodetr_baseline"
    echo "training_split=trainval_7481"
    echo "hidden_test_split=test_7518"
    echo "batch_size=12"
    echo "fixed_epoch=195"
    echo "strong_control=false"
    echo "qlqc=false"
    echo "test_tuning=forbidden"
    echo "created=$(date '+%F %T')"
    echo "project_root=${PROJECT_ROOT}"
    sha256sum "${TRAIN_CONFIG}" "${EVAL_CONFIG}"
    python --version
    python - <<'PY'
import cv2
import torch

print("opencv={}".format(cv2.__version__))
print("pytorch={}".format(torch.__version__))
print("cuda_runtime={}".format(torch.version.cuda))
print("gpu={}".format(torch.cuda.get_device_name(0)))
PY
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  } > "${PROTOCOL_FILE}"
}

run_training() {
  if checkpoint_is_valid "${SOURCE_CHECKPOINT}" 195 >/dev/null 2>&1; then
    echo "195轮StereoDETR同环境基线已经完成，跳过训练。"
    checkpoint_is_valid "${SOURCE_CHECKPOINT}" 195
    return
  fi

  mkdir -p "$(dirname "${TRAIN_LOG}")"
  {
    echo "stage=StereoDETR同环境trainval基线"
    echo "config=${TRAIN_CONFIG}"
    echo "start=$(date '+%F %T')"
    echo "resume_if_exists=true"
    echo "expected_final_epoch=195"
  } | tee -a "${TRAIN_INFO}"

  local start_ts
  start_ts="$(date +%s)"
  set +e
  python tools/训练与评估_无蒸馏QECR.py \
    --config "${TRAIN_CONFIG}" \
    2>&1 | tee -a "${TRAIN_LOG}"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  local elapsed="$(( $(date +%s) - start_ts ))"
  {
    echo "end=$(date '+%F %T')"
    echo "exit_code=${exit_code}"
    echo "elapsed_seconds=${elapsed}"
  } | tee -a "${TRAIN_INFO}" "${TRAIN_LOG}"

  [[ "${exit_code}" -eq 0 ]] || die "基线训练失败，退出码=${exit_code}。修复后重新运行train或all会断点续跑"
  checkpoint_is_valid "${SOURCE_CHECKPOINT}" 195
}

count_results() {
  if [[ -d "${RESULT_DIR}" ]]; then
    find "${RESULT_DIR}" -maxdepth 1 -type f -name '*.txt' | wc -l
  else
    echo 0
  fi
}

run_inference() {
  checkpoint_is_valid "${SOURCE_CHECKPOINT}" 195

  if [[ -f "${SUBMISSION_ZIP}" ]]; then
    unzip -t "${SUBMISSION_ZIP}" >/dev/null
    echo "基线提交包已经存在且ZIP完整，本次不重复推理：${SUBMISSION_ZIP}"
    return
  fi

  mkdir -p "${EVAL_DIR}" "${EVAL_ROOT}"
  local source_sha
  source_sha="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
  if [[ -f "${TARGET_CHECKPOINT}" ]]; then
    local target_sha
    target_sha="$(sha256sum "${TARGET_CHECKPOINT}" | awk '{print $1}')"
    [[ "${source_sha}" == "${target_sha}" ]] || die "官网推理目录中的checkpoint_best.pth不是当前T0固定权重"
  else
    cp "${SOURCE_CHECKPOINT}" "${TARGET_CHECKPOINT}"
  fi

  local result_count
  result_count="$(count_results | tr -d '[:space:]')"
  if [[ "${result_count}" == "7518" ]]; then
    echo "检测到完整的7518个基线结果文件，直接执行格式审计和打包。"
  elif [[ "${result_count}" != "0" ]]; then
    die "基线结果目录只有${result_count}个txt。请先人工备份并移走不完整结果目录，再重新运行infer"
  else
    {
      echo "purpose=single_stereodetr_reimpl_baseline_KITTI_test_inference"
      echo "training=none"
      echo "config=${EVAL_CONFIG}"
      echo "source_checkpoint=${SOURCE_CHECKPOINT}"
      echo "source_checkpoint_sha256=${source_sha}"
      echo "start=$(date '+%F %T')"
    } > "${EVAL_INFO}"

    local start_ts
    start_ts="$(date +%s)"
    set +e
    python tools/训练与评估_无蒸馏QECR.py \
      --evaluate_only \
      --config "${EVAL_CONFIG}" \
      2>&1 | tee "${EVAL_LOG}"
    local exit_code="${PIPESTATUS[0]}"
    set -e
    {
      echo "end=$(date '+%F %T')"
      echo "exit_code=${exit_code}"
      echo "elapsed_seconds=$(( $(date +%s) - start_ts ))"
    } | tee -a "${EVAL_INFO}" "${EVAL_LOG}"
    [[ "${exit_code}" -eq 0 ]] || die "KITTI test基线推理失败，退出码=${exit_code}"
  fi

  python tools/检查并打包KITTI提交.py \
    --result_dir "${RESULT_DIR}" \
    --output_zip "${SUBMISSION_ZIP}"
  echo "StereoDETR同环境基线官网提交包：${PROJECT_ROOT}/${SUBMISSION_ZIP}"
}

show_status() {
  echo "===== StereoDETR同环境基线状态 ====="
  if checkpoint_is_valid "${SOURCE_CHECKPOINT}" 195 >/dev/null 2>&1; then
    echo "训练完成：${SOURCE_CHECKPOINT}"
  elif [[ -f "${TRAIN_DIR}/checkpoint.pth" ]]; then
    echo "存在滚动checkpoint，可断点续跑：${TRAIN_DIR}/checkpoint.pth"
  else
    echo "尚未开始195轮基线训练"
  fi
  echo "KITTI基线结果文件数：$(count_results | tr -d '[:space:]') / 7518"
  if [[ -f "${SUBMISSION_ZIP}" ]]; then
    ls -lh "${SUBMISSION_ZIP}" "${SUBMISSION_ZIP}.sha256"
  else
    echo "基线提交包尚未生成"
  fi
}

require_environment

if [[ "${MODE}" == "status" ]]; then
  show_status
  exit 0
fi

run_preflight
write_protocol_record

if [[ "${MODE}" == "preflight" ]]; then
  exit 0
fi

if [[ "${MODE}" == "all" || "${MODE}" == "train" ]]; then
  run_training
fi

if [[ "${MODE}" == "all" || "${MODE}" == "infer" ]]; then
  run_inference
fi

show_status
