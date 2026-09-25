#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/运行官方批量16_QLQC最终提交.sh MODE PROFILE
# MODE: preflight | smoke | train_base | infer_base | train_qlqc |
#       infer_qlqc | all | status
# PROFILE: single_fp32 | single_amp | dual_fp32

MODE="${1:-preflight}"
PROFILE="${2:-single_fp32}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

case "${MODE}" in
  preflight|smoke|train_base|infer_base|train_qlqc|infer_qlqc|all|status)
    ;;
  *)
    echo "错误：未知模式 ${MODE}" >&2
    echo "用法：bash scripts/运行官方批量16_QLQC最终提交.sh [preflight|smoke|train_base|infer_base|train_qlqc|infer_qlqc|all|status] [single_fp32|single_amp|dual_fp32]" >&2
    exit 2
    ;;
esac

case "${PROFILE}" in
  single_fp32)
    VISIBLE_GPUS="${QECR_CUDA_VISIBLE_DEVICES:-0}"
    RUNTIME_GPU_IDS="0"
    REQUIRED_GPU_COUNT=1
    AMP_ARGUMENT="--no_amp"
    ;;
  single_amp)
    VISIBLE_GPUS="${QECR_CUDA_VISIBLE_DEVICES:-0}"
    RUNTIME_GPU_IDS="0"
    REQUIRED_GPU_COUNT=1
    AMP_ARGUMENT="--amp"
    ;;
  dual_fp32)
    VISIBLE_GPUS="${QECR_CUDA_VISIBLE_DEVICES:-0,1}"
    RUNTIME_GPU_IDS="0,1"
    REQUIRED_GPU_COUNT=2
    AMP_ARGUMENT="--no_amp"
    ;;
  *)
    echo "错误：未知运行档位 ${PROFILE}" >&2
    exit 2
    ;;
esac

TRAIN_ROOT="/root/autodl-tmp/datasets/KITTI/object/training"
TEST_ROOT="/root/autodl-tmp/datasets/KITTI/object/testing"
TRAINVAL_LIST="${TRAIN_ROOT}/ImageSets/trainval.txt"
TEST_LIST="${TEST_ROOT}/ImageSets/test.txt"
SMOKE_LIST="${TRAIN_ROOT}/ImageSets/train_batch16_smoke.txt"

CONFIG_ROOT="configs/官方批量16复核"
T0_CONFIG="${CONFIG_ROOT}/T0_trainval_StereoDETR.yaml"
T1_CONFIG="${CONFIG_ROOT}/T1_trainval强控制.yaml"
T2_CONFIG="${CONFIG_ROOT}/T2_trainval_QLQC.yaml"
E0_CONFIG="${CONFIG_ROOT}/E0_StereoDETR_KITTI测试.yaml"
E2_CONFIG="${CONFIG_ROOT}/E2_QLQC_KITTI测试.yaml"
SMOKE_CONFIG="${CONFIG_ROOT}/批量16显存冒烟.yaml"

RUN_ROOT="outputs/KITTI_trainval官方批量16复核"
T0_DIR="${RUN_ROOT}/T0_StereoDETR/stereodetr_trainval_global_batch16"
T1_DIR="${RUN_ROOT}/T1_强控制/stereodetr_trainval_global_batch16_strong_control"
T2_DIR="${RUN_ROOT}/T2_QLQC/qlqc_trainval_global_batch16"
E0_DIR="${RUN_ROOT}/官网测试/StereoDETR/stereodetr_global_batch16_kitti_test"
E2_DIR="${RUN_ROOT}/官网测试/QLQC/qlqc_global_batch16_kitti_test"
E0_ZIP="${RUN_ROOT}/StereoDETR_global_batch16_KITTI_test_submission.zip"
E2_ZIP="${RUN_ROOT}/QLQC_global_batch16_KITTI_test_submission.zip"
PROFILE_LOCK="${RUN_ROOT}/训练运行档位.txt"
PROTOCOL_FILE="${RUN_ROOT}/固定协议与环境.txt"

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
  export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export PYTHONUNBUFFERED=1
}

validate_visible_gpus() {
  python - "${REQUIRED_GPU_COUNT}" "${PROFILE}" <<'PY'
import sys
import torch

required = int(sys.argv[1])
profile = sys.argv[2]
if not torch.cuda.is_available():
    raise SystemExit("CUDA不可用")
available = torch.cuda.device_count()
if available < required:
    raise SystemExit(
        f"{profile}需要{required}张可见GPU，当前只有{available}张"
    )
print(f"运行档位：{profile}")
print(f"可见GPU数：{available}")
for index in range(available):
    properties = torch.cuda.get_device_properties(index)
    print(
        "GPU{}: {}，显存{:.2f} GiB".format(
            index,
            properties.name,
            properties.total_memory / 1024 ** 3,
        )
    )
PY
}

validate_datasets() {
  mkdir -p "${TEST_ROOT}/ImageSets"
  if [[ ! -f "${TEST_LIST}" ]]; then
    seq -f '%06g' 0 7517 > "${TEST_LIST}"
  fi
  python - "${TRAIN_ROOT}" "${TRAINVAL_LIST}" "${TEST_ROOT}" "${TEST_LIST}" <<'PY'
from pathlib import Path
import sys

train_root = Path(sys.argv[1])
trainval_file = Path(sys.argv[2])
test_root = Path(sys.argv[3])
test_file = Path(sys.argv[4])

if not trainval_file.is_file():
    raise SystemExit(f"缺少trainval列表：{trainval_file}")
train_ids = [line.strip() for line in trainval_file.read_text().splitlines() if line.strip()]
if sorted(train_ids) != [f"{index:06d}" for index in range(7481)]:
    raise SystemExit("trainval.txt必须无重复覆盖000000到007480")
test_ids = [line.strip() for line in test_file.read_text().splitlines() if line.strip()]
if test_ids != [f"{index:06d}" for index in range(7518)]:
    raise SystemExit("test.txt必须按顺序覆盖000000到007517")

requirements = [
    (train_root, train_ids, (("calib", ".txt"), ("image_2", ".png"),
                             ("image_3", ".png"), ("label_2", ".txt"))),
    (test_root, test_ids, (("calib", ".txt"), ("image_2", ".png"),
                           ("image_3", ".png"))),
]
missing = []
for root, sample_ids, folders in requirements:
    for sample_id in sample_ids:
        for folder, suffix in folders:
            path = root / folder / f"{sample_id}{suffix}"
            if not path.is_file():
                missing.append(str(path))
                if len(missing) >= 20:
                    raise SystemExit("KITTI数据不完整：\n" + "\n".join(missing))
if missing:
    raise SystemExit("KITTI数据不完整：\n" + "\n".join(missing))
print("KITTI数据通过：trainval 7481张，test 7518张")
PY
}

validate_final_checkpoint() {
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
        f"{path}轮次为{checkpoint.get('epoch')}，期望{expected_epoch}"
    )
state = checkpoint.get("model_state")
if not isinstance(state, dict) or not state:
    raise SystemExit(f"{path}不含有效model_state")
print(f"checkpoint通过：{path}，epoch={expected_epoch}，tensor={len(state)}")
PY
}

run_preflight() {
  python tools/检查官方批量16训练协议.py
  python tools/测试固定轮次Checkpoint.py
  python tools/测试Checkpoint兼容白名单.py
  python tools/测试3D质量排序.py
  python tools/测试AMP匹配器.py
  python -m py_compile \
    lib/helpers/parallel_helper.py \
    lib/helpers/trainer_helper.py \
    lib/helpers/save_helper.py \
    lib/models/monodetr/ops/functions/ms_deform_attn_func.py \
    tools/训练与评估_QECR.py \
    tools/测试AMP匹配器.py
  validate_visible_gpus
  validate_datasets
  echo "官方批量16训练预检通过：PROFILE=${PROFILE}，global_batch=16"
}

write_protocol_record() {
  mkdir -p "${RUN_ROOT}"
  {
    echo "protocol=trainval_7481_fixed_195_plus_1_plus_3"
    echo "global_batch_size=16"
    echo "runtime_profile=${PROFILE}"
    echo "visible_physical_gpus=${VISIBLE_GPUS}"
    echo "runtime_gpu_ids=${RUNTIME_GPU_IDS}"
    echo "amp=$([[ "${AMP_ARGUMENT}" == "--amp" ]] && echo true || echo false)"
    echo "selection_split=Chen_val_3769"
    echo "hidden_test_tuning=forbidden"
    echo "created=$(date '+%F %T')"
    sha256sum "${T0_CONFIG}" "${T1_CONFIG}" "${T2_CONFIG}" "${E0_CONFIG}" "${E2_CONFIG}"
    python --version
    python - <<'PY'
import cv2
import torch
print("opencv={}".format(cv2.__version__))
print("pytorch={}".format(torch.__version__))
print("cuda_runtime={}".format(torch.version.cuda))
PY
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  } > "${PROTOCOL_FILE}"
}

lock_training_profile() {
  mkdir -p "${RUN_ROOT}"
  local signature="profile=${PROFILE};visible=${VISIBLE_GPUS};global_batch=16;amp=${AMP_ARGUMENT}"
  if [[ -f "${PROFILE_LOCK}" ]]; then
    local existing
    existing="$(tr -d '\r\n' < "${PROFILE_LOCK}")"
    if [[ "${existing}" != "${signature}" ]]; then
      if [[ -f "${T0_DIR}/checkpoint.pth" \
            || -f "${T0_DIR}/checkpoint_final.pth" \
            || -f "${T1_DIR}/checkpoint.pth" \
            || -f "${T1_DIR}/checkpoint_final.pth" \
            || -f "${T2_DIR}/checkpoint.pth" \
            || -f "${T2_DIR}/checkpoint_final.pth" ]]; then
        die "已有训练使用 ${existing}，不能与 ${signature} 混合续跑"
      fi
      echo "${signature}" > "${PROFILE_LOCK}"
      echo "尚无正式训练checkpoint，已将运行档位改为 ${PROFILE}"
    fi
  else
    echo "${signature}" > "${PROFILE_LOCK}"
  fi
}

runtime_arguments() {
  RUNTIME_ARGUMENTS=(
    --batch_size 16
    --gpu_ids "${RUNTIME_GPU_IDS}"
    "${AMP_ARGUMENT}"
  )
}

run_smoke() {
  mkdir -p "$(dirname "${SMOKE_LIST}")" "${RUN_ROOT}/批量16显存冒烟"
  head -n 32 "${TRAINVAL_LIST}" > "${SMOKE_LIST}"
  local log_file="${RUN_ROOT}/批量16显存冒烟/${PROFILE}.log"
  runtime_arguments
  set +e
  python tools/训练与评估_无蒸馏QECR.py \
    --config "${SMOKE_CONFIG}" \
    "${RUNTIME_ARGUMENTS[@]}" \
    2>&1 | tee "${log_file}"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  if [[ "${exit_code}" -ne 0 ]]; then
    die "${PROFILE}批量16冒烟失败，退出码=${exit_code}，见${log_file}"
  fi
  grep -q "non-finite detector loss" "${log_file}" && die "冒烟出现非有限损失"
  echo "批量16冒烟通过：${PROFILE}，完成2次前向、反向和优化更新"
}

run_training_stage() {
  local stage_name="$1"
  local config_path="$2"
  local model_dir="$3"
  local expected_epoch="$4"
  local final_checkpoint="${model_dir}/checkpoint_final.pth"
  local stage_root
  stage_root="$(dirname "${model_dir}")"
  local log_file="${stage_root}/控制台.log"
  local info_file="${stage_root}/运行信息.txt"

  if validate_final_checkpoint "${final_checkpoint}" "${expected_epoch}" >/dev/null 2>&1; then
    echo "${stage_name}已经完成，跳过训练"
    validate_final_checkpoint "${final_checkpoint}" "${expected_epoch}"
    return
  fi

  mkdir -p "${stage_root}"
  {
    echo "stage=${stage_name}"
    echo "config=${config_path}"
    echo "runtime_profile=${PROFILE}"
    echo "global_batch_size=16"
    echo "start=$(date '+%F %T')"
    echo "resume_if_exists=true"
  } | tee -a "${info_file}"

  runtime_arguments
  local start_ts
  start_ts="$(date +%s)"
  set +e
  python tools/训练与评估_无蒸馏QECR.py \
    --config "${config_path}" \
    "${RUNTIME_ARGUMENTS[@]}" \
    2>&1 | tee -a "${log_file}"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  local elapsed="$(( $(date +%s) - start_ts ))"
  {
    echo "end=$(date '+%F %T')"
    echo "exit_code=${exit_code}"
    echo "elapsed_seconds=${elapsed}"
  } | tee -a "${info_file}" "${log_file}"
  [[ "${exit_code}" -eq 0 ]] || die "${stage_name}失败，修复后用相同PROFILE重跑可从checkpoint.pth续训"
  validate_final_checkpoint "${final_checkpoint}" "${expected_epoch}"
}

run_base_training() {
  lock_training_profile
  write_protocol_record
  run_training_stage "T0_StereoDETR_batch16" "${T0_CONFIG}" "${T0_DIR}" 195
}

run_qlqc_training() {
  validate_final_checkpoint "${T0_DIR}/checkpoint_final.pth" 195 >/dev/null || die "请先完成train_base"
  lock_training_profile
  write_protocol_record
  run_training_stage "T1_强控制_batch16" "${T1_CONFIG}" "${T1_DIR}" 1
  run_training_stage "T2_QLQC_batch16" "${T2_CONFIG}" "${T2_DIR}" 3
}

count_txt() {
  local result_dir="$1"
  if [[ -d "${result_dir}" ]]; then
    find "${result_dir}" -maxdepth 1 -type f -name '*.txt' | wc -l
  else
    echo 0
  fi
}

run_inference_and_package() {
  local label="$1"
  local config_path="$2"
  local source_checkpoint="$3"
  local expected_epoch="$4"
  local eval_dir="$5"
  local zip_path="$6"
  local target_checkpoint="${eval_dir}/checkpoint_best.pth"
  local result_dir="${eval_dir}/outputs/data"
  local log_file="$(dirname "${eval_dir}")/控制台.log"

  validate_final_checkpoint "${source_checkpoint}" "${expected_epoch}"
  if [[ -f "${zip_path}" ]]; then
    unzip -t "${zip_path}" >/dev/null
    echo "${label}提交包已存在且完整：${zip_path}"
    return
  fi
  local result_count
  result_count="$(count_txt "${result_dir}" | tr -d '[:space:]')"
  if [[ "${result_count}" != "0" && "${result_count}" != "7518" ]]; then
    die "${label}结果目录含${result_count}/7518个txt，请先人工备份并移走该不完整目录"
  fi
  if [[ "${result_count}" == "0" ]]; then
    mkdir -p "${eval_dir}" "$(dirname "${log_file}")"
    if [[ -f "${target_checkpoint}" ]]; then
      [[ "$(sha256sum "${source_checkpoint}" | awk '{print $1}')" == "$(sha256sum "${target_checkpoint}" | awk '{print $1}')" ]] \
        || die "${label}推理目录中的checkpoint_best.pth与固定训练权重不同"
    else
      cp "${source_checkpoint}" "${target_checkpoint}"
    fi
    set +e
    python tools/训练与评估_无蒸馏QECR.py \
      --evaluate_only \
      --config "${config_path}" \
      --batch_size 12 \
      --gpu_ids 0 \
      --no_amp \
      2>&1 | tee "${log_file}"
    local exit_code="${PIPESTATUS[0]}"
    set -e
    [[ "${exit_code}" -eq 0 ]] || die "${label} KITTI test推理失败"
  fi
  python tools/检查并打包KITTI提交.py \
    --result_dir "${result_dir}" \
    --output_zip "${zip_path}"
  echo "${label}提交包：${PROJECT_ROOT}/${zip_path}"
}

infer_base() {
  run_inference_and_package \
    "StereoDETR批量16基线" \
    "${E0_CONFIG}" \
    "${T0_DIR}/checkpoint_final.pth" \
    195 \
    "${E0_DIR}" \
    "${E0_ZIP}"
}

infer_qlqc() {
  run_inference_and_package \
    "QLQC批量16最终模型" \
    "${E2_CONFIG}" \
    "${T2_DIR}/checkpoint_final.pth" \
    3 \
    "${E2_DIR}" \
    "${E2_ZIP}"
}

show_status() {
  echo "===== 官方批量16复核状态 ====="
  [[ -f "${PROFILE_LOCK}" ]] && cat "${PROFILE_LOCK}" || echo "训练档位尚未锁定"
  for item in \
    "${T0_DIR}/checkpoint_final.pth:195:T0基础模型" \
    "${T1_DIR}/checkpoint_final.pth:1:T1强控制" \
    "${T2_DIR}/checkpoint_final.pth:3:T2 QLQC"
  do
    checkpoint="${item%%:*}"
    rest="${item#*:}"
    epoch="${rest%%:*}"
    label="${rest#*:}"
    if validate_final_checkpoint "${checkpoint}" "${epoch}" >/dev/null 2>&1; then
      echo "完成  ${label}  epoch=${epoch}"
    elif [[ -f "$(dirname "${checkpoint}")/checkpoint.pth" ]]; then
      echo "续跑中  ${label}"
    else
      echo "未开始  ${label}"
    fi
  done
  echo "StereoDETR test结果：$(count_txt "${E0_DIR}/outputs/data" | tr -d '[:space:]') / 7518"
  echo "QLQC test结果：$(count_txt "${E2_DIR}/outputs/data" | tr -d '[:space:]') / 7518"
  [[ -f "${E0_ZIP}" ]] && ls -lh "${E0_ZIP}" "${E0_ZIP}.sha256" || true
  [[ -f "${E2_ZIP}" ]] && ls -lh "${E2_ZIP}" "${E2_ZIP}.sha256" || true
}

require_environment

if [[ "${MODE}" == "status" ]]; then
  show_status
  exit 0
fi

run_preflight

case "${MODE}" in
  preflight)
    ;;
  smoke)
    run_smoke
    ;;
  train_base)
    run_base_training
    ;;
  infer_base)
    infer_base
    ;;
  train_qlqc)
    run_qlqc_training
    ;;
  infer_qlqc)
    infer_qlqc
    ;;
  all)
    run_base_training
    infer_base
    run_qlqc_training
    infer_qlqc
    ;;
esac

show_status
