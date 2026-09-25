#!/usr/bin/env bash
set -euo pipefail

# 固定协议：7481张有标签图像只用于训练，7518张KITTI test只推理一次。
# 阶段和轮数均由既有Chen划分实验预先确定，不根据隐藏test结果调参。

MODE="${1:-all}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_ROOT="/root/autodl-tmp/datasets/KITTI/object/training"
TEST_ROOT="/root/autodl-tmp/datasets/KITTI/object/testing"
TRAINVAL_LIST="${TRAIN_ROOT}/ImageSets/trainval.txt"
TEST_LIST="${TEST_ROOT}/ImageSets/test.txt"

T0_CONFIG="configs/V09_trainval官网复核/V09T0_trainval基础模型.yaml"
T1_CONFIG="configs/V09_trainval官网复核/V09T1_trainval强控制.yaml"
T2_CONFIG="configs/V09_trainval官网复核/V09T2_trainval_QLQC.yaml"
E2_CONFIG="configs/V09_trainval官网复核/V09E2_QLQC官网测试.yaml"

RUN_ROOT="outputs/KITTI_trainval官网复核"
T0_DIR="${RUN_ROOT}/V09T0_trainval基础模型/qecr_v09t0_trainval_stereodetr"
T1_DIR="${RUN_ROOT}/V09T1_trainval强控制/qecr_v09t1_trainval_strong_control"
T2_DIR="${RUN_ROOT}/V09T2_trainval_QLQC/qecr_v09t2_trainval_qlqc"
E2_DIR="${RUN_ROOT}/官网测试/QLQC最终模型/qecr_v09e2_trainval_qlqc_kitti_test"
RESULT_DIR="${E2_DIR}/outputs/data"
SUBMISSION_ZIP="${RUN_ROOT}/V09_trainval_QLQC_KITTI_test_submission.zip"
PROTOCOL_FILE="${RUN_ROOT}/固定协议与环境.txt"

case "${MODE}" in
  all|preflight|train|infer|status)
    ;;
  *)
    echo "用法：bash scripts/运行V09_trainval官网最终提交.sh [all|preflight|train|infer|status]" >&2
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
expected_train = [f"{index:06d}" for index in range(7481)]
if sorted(train_ids) != expected_train or len(set(train_ids)) != 7481:
    raise SystemExit("trainval.txt必须无重复地覆盖000000到007480共7481个训练样本")

test_ids = [line.strip() for line in test_file.read_text().splitlines() if line.strip()]
expected_test = [f"{index:06d}" for index in range(7518)]
if test_ids != expected_test:
    raise SystemExit("test.txt必须按顺序覆盖000000到007517共7518个测试样本")

missing = []
for sample_id in train_ids:
    for folder, suffix in (
        ("calib", ".txt"),
        ("image_2", ".png"),
        ("image_3", ".png"),
        ("label_2", ".txt"),
    ):
        path = train_root / folder / f"{sample_id}{suffix}"
        if not path.is_file():
            missing.append(str(path))
            if len(missing) >= 20:
                break
    if len(missing) >= 20:
        break

for sample_id in test_ids:
    for folder, suffix in (
        ("calib", ".txt"),
        ("image_2", ".png"),
        ("image_3", ".png"),
    ):
        path = test_root / folder / f"{sample_id}{suffix}"
        if not path.is_file():
            missing.append(str(path))
            if len(missing) >= 20:
                break
    if len(missing) >= 20:
        break

if missing:
    raise SystemExit("KITTI数据不完整，示例缺失文件：\n" + "\n".join(missing))

print("KITTI数据检查通过：trainval 7481张有标签双目图，test 7518张无标签双目图")
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
        f"{path}轮次异常：{checkpoint.get('epoch')}，期望{expected_epoch}"
    )
state = checkpoint.get("model_state")
if not isinstance(state, dict) or not state:
    raise SystemExit(f"{path}不含有效model_state")
print(f"固定轮次checkpoint通过：{path}，epoch={expected_epoch}，tensor={len(state)}")
PY
}

write_protocol_record() {
  mkdir -p "${RUN_ROOT}"
  {
    echo "protocol=V09_trainval_fixed_epoch_then_single_KITTI_test"
    echo "selection_split=Chen_val_3769"
    echo "final_training_split=trainval_7481"
    echo "hidden_test_split=test_7518"
    echo "fixed_epochs=195+1+3"
    echo "test_tuning=forbidden"
    echo "created=$(date '+%F %T')"
    echo "project_root=${PROJECT_ROOT}"
    echo "python=$(command -v python)"
    sha256sum "${T0_CONFIG}" "${T1_CONFIG}" "${T2_CONFIG}" "${E2_CONFIG}"
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

run_training_stage() {
  local stage_name="$1"
  local config_path="$2"
  local model_dir="$3"
  local expected_epoch="$4"
  local final_checkpoint="${model_dir}/checkpoint_final.pth"
  local log_file="$(dirname "${model_dir}")/控制台.log"
  local info_file="$(dirname "${model_dir}")/运行信息.txt"

  if validate_final_checkpoint "${final_checkpoint}" "${expected_epoch}" >/dev/null 2>&1; then
    echo "${stage_name}已经完成，跳过训练。"
    validate_final_checkpoint "${final_checkpoint}" "${expected_epoch}"
    return
  fi

  mkdir -p "$(dirname "${model_dir}")"
  {
    echo "stage=${stage_name}"
    echo "config=${config_path}"
    echo "start=$(date '+%F %T')"
    echo "resume_if_exists=true"
    echo "expected_final_epoch=${expected_epoch}"
  } | tee -a "${info_file}"

  local start_ts
  start_ts="$(date +%s)"
  set +e
  python tools/训练与评估_无蒸馏QECR.py \
    --config "${config_path}" \
    2>&1 | tee -a "${log_file}"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  local elapsed="$(( $(date +%s) - start_ts ))"
  {
    echo "end=$(date '+%F %T')"
    echo "exit_code=${exit_code}"
    echo "elapsed_seconds=${elapsed}"
  } | tee -a "${info_file}" "${log_file}"

  if [[ "${exit_code}" -ne 0 ]]; then
    die "${stage_name}失败，退出码=${exit_code}。修复后重新运行会从checkpoint.pth续跑"
  fi
  validate_final_checkpoint "${final_checkpoint}" "${expected_epoch}"
}

run_train_chain() {
  run_training_stage "V09T0_trainval基础模型" "${T0_CONFIG}" "${T0_DIR}" 195
  run_training_stage "V09T1_trainval强控制" "${T1_CONFIG}" "${T1_DIR}" 1
  run_training_stage "V09T2_trainval_QLQC" "${T2_CONFIG}" "${T2_DIR}" 3
}

count_result_files() {
  if [[ -d "${RESULT_DIR}" ]]; then
    find "${RESULT_DIR}" -maxdepth 1 -type f -name '*.txt' | wc -l
  else
    echo 0
  fi
}

run_final_inference() {
  local source_checkpoint="${T2_DIR}/checkpoint_final.pth"
  local target_checkpoint="${E2_DIR}/checkpoint_best.pth"
  local log_file="${RUN_ROOT}/官网测试/QLQC最终模型/控制台.log"
  local info_file="${RUN_ROOT}/官网测试/QLQC最终模型/运行信息.txt"

  validate_final_checkpoint "${source_checkpoint}" 3

  if [[ -f "${SUBMISSION_ZIP}" ]]; then
    unzip -t "${SUBMISSION_ZIP}" >/dev/null
    echo "最终提交包已经存在且ZIP完整，本次不重复推理：${SUBMISSION_ZIP}"
    return
  fi

  local result_count
  result_count="$(count_result_files | tr -d '[:space:]')"
  if [[ "${result_count}" == "7518" ]]; then
    echo "检测到完整的7518个结果文件，直接执行格式审计和打包。"
  elif [[ "${result_count}" != "0" ]]; then
    die "结果目录只有${result_count}个txt。请先人工备份并移走该不完整目录，再重跑infer"
  else
    mkdir -p "${E2_DIR}" "$(dirname "${log_file}")"
    if [[ -f "${target_checkpoint}" ]]; then
      local source_sha target_sha
      source_sha="$(sha256sum "${source_checkpoint}" | awk '{print $1}')"
      target_sha="$(sha256sum "${target_checkpoint}" | awk '{print $1}')"
      [[ "${source_sha}" == "${target_sha}" ]] || die "官网推理目录中的checkpoint_best.pth不是本次T2固定权重"
    else
      cp "${source_checkpoint}" "${target_checkpoint}"
    fi

    {
      echo "purpose=single_final_KITTI_test_inference"
      echo "training=none"
      echo "config=${E2_CONFIG}"
      echo "checkpoint=${source_checkpoint}"
      echo "checkpoint_sha256=$(sha256sum "${source_checkpoint}" | awk '{print $1}')"
      echo "start=$(date '+%F %T')"
    } > "${info_file}"

    local start_ts
    start_ts="$(date +%s)"
    set +e
    python tools/训练与评估_无蒸馏QECR.py \
      --evaluate_only \
      --config "${E2_CONFIG}" \
      2>&1 | tee "${log_file}"
    local exit_code="${PIPESTATUS[0]}"
    set -e
    {
      echo "end=$(date '+%F %T')"
      echo "exit_code=${exit_code}"
      echo "elapsed_seconds=$(( $(date +%s) - start_ts ))"
    } | tee -a "${info_file}" "${log_file}"
    [[ "${exit_code}" -eq 0 ]] || die "KITTI test推理失败，退出码=${exit_code}"
  fi

  python tools/检查并打包KITTI提交.py \
    --result_dir "${RESULT_DIR}" \
    --output_zip "${SUBMISSION_ZIP}"
  echo "最终且唯一的官网提交包：${PROJECT_ROOT}/${SUBMISSION_ZIP}"
}

show_status() {
  echo "===== 固定轮次checkpoint ====="
  for item in \
    "${T0_DIR}/checkpoint_final.pth:195" \
    "${T1_DIR}/checkpoint_final.pth:1" \
    "${T2_DIR}/checkpoint_final.pth:3"
  do
    checkpoint="${item%:*}"
    epoch="${item##*:}"
    if validate_final_checkpoint "${checkpoint}" "${epoch}" >/dev/null 2>&1; then
      echo "完成  epoch=${epoch}  ${checkpoint}"
    elif [[ -f "$(dirname "${checkpoint}")/checkpoint.pth" ]]; then
      echo "续跑中  ${checkpoint}"
    else
      echo "未开始  ${checkpoint}"
    fi
  done
  echo "KITTI结果文件数：$(count_result_files | tr -d '[:space:]') / 7518"
  if [[ -f "${SUBMISSION_ZIP}" ]]; then
    ls -lh "${SUBMISSION_ZIP}" "${SUBMISSION_ZIP}.sha256"
  else
    echo "提交包尚未生成"
  fi
}

require_environment

if [[ "${MODE}" == "status" ]]; then
  show_status
  exit 0
fi

python tools/检查V09Trainval官网复核配置.py
python tools/测试固定轮次Checkpoint.py
python tools/测试Checkpoint兼容白名单.py
python tools/测试3D质量排序.py
validate_datasets

if [[ "${MODE}" == "preflight" ]]; then
  echo "V09 trainval官网最终提交预检通过"
  exit 0
fi

write_protocol_record

if [[ "${MODE}" == "all" || "${MODE}" == "train" ]]; then
  run_train_chain
fi

if [[ "${MODE}" == "all" || "${MODE}" == "infer" ]]; then
  run_final_inference
fi

show_status
