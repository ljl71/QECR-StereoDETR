#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pair}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SOURCE_CHECKPOINT="${PROJECT_ROOT}/outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
RUN_ROOT="outputs/第三阶段_V20立体几何深度残差"
CONTROL_CONFIG="versions/V20O_V09零训练深度控制/config.yaml"
QUERY_CONFIG="versions/V20A_查询级深度残差/config.yaml"
GEOMETRY_CONFIG="versions/V20B_立体几何深度残差/config.yaml"
SMOKE_CONFIG="configs/AutoDL_V20B_立体几何深度残差冒烟.yaml"
CONTROL_DIR="${RUN_ROOT}/V20O_V09零训练深度控制"
QUERY_DIR="${RUN_ROOT}/V20A_查询级深度残差"
GEOMETRY_DIR="${RUN_ROOT}/V20B_立体几何深度残差"
SMOKE_DIR="${RUN_ROOT}/冒烟测试"
CONTROL_MODEL="qecr_v20o_v09_zero_train_depth_control"
QUERY_MODEL="qecr_v20a_query_only_depth_residual"
GEOMETRY_MODEL="qecr_v20b_stereo_geometry_depth_residual"
SMOKE_MODEL="qecr_v20b_geometry_depth_residual_smoke"
CONTROL_LOG="${CONTROL_DIR}/控制台.log"
QUERY_LOG="${QUERY_DIR}/控制台.log"
GEOMETRY_LOG="${GEOMETRY_DIR}/控制台.log"
SMOKE_LOG="${SMOKE_DIR}/控制台.log"
REPORT="${RUN_ROOT}/V20配对结果.md"

case "${MODE}" in
  smoke|pair|control|query|geometry|summary) ;;
  *)
    echo "用法：bash scripts/运行V20立体几何深度残差.sh [smoke|pair|control|query|geometry|summary]" >&2
    exit 2
    ;;
esac

if [[ "${CONDA_DEFAULT_ENV:-}" != "stereodetr-open" ]]; then
  echo "错误：请先激活stereodetr-open环境。" >&2
  exit 2
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NUMBA_CUDA_USE_NVIDIA_BINDING="${NUMBA_CUDA_USE_NVIDIA_BINDING:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "错误：没有找到V09最佳权重：${SOURCE_CHECKPOINT}" >&2
  exit 4
fi

preflight() {
  python tools/测试V20立体几何深度残差.py
  python tools/汇总V20立体几何深度残差.py --self_test
}

ensure_empty() {
  local stage_dir="$1"
  if [[ -d "${stage_dir}" ]] && [[ -n "$(find "${stage_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "错误：${stage_dir}已经包含实验产物；为防止覆盖，本次未启动。" >&2
    exit 3
  fi
}

run_control() {
  ensure_empty "${CONTROL_DIR}"
  local checkpoint_dir="${CONTROL_DIR}/${CONTROL_MODEL}"
  local checkpoint_path="${checkpoint_dir}/checkpoint_best.pth"
  mkdir -p "${checkpoint_dir}"
  cp "${CONTROL_CONFIG}" "${CONTROL_DIR}/配置快照.yaml"
  cp --reflink=auto "${SOURCE_CHECKPOINT}" "${checkpoint_path}"
  local source_sha copied_sha
  source_sha="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
  copied_sha="$(sha256sum "${checkpoint_path}" | awk '{print $1}')"
  [[ "${source_sha}" == "${copied_sha}" ]] || {
    echo "错误：V20O复制后的checkpoint校验失败。" >&2
    exit 5
  }
  {
    echo "stage=V20O_V09零训练深度控制"
    echo "training=none"
    echo "source_sha256=${source_sha}"
    echo "copied_sha256=${copied_sha}"
    echo "start=$(date '+%F %T')"
  } > "${CONTROL_DIR}/运行信息.txt"
  set +e
  python tools/训练与评估_无蒸馏QECR.py \
    --evaluate_only --config "${CONTROL_CONFIG}" \
    2>&1 | tee "${CONTROL_LOG}"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  echo "exit_code=${exit_code}" | tee -a "${CONTROL_DIR}/运行信息.txt" "${CONTROL_LOG}"
  [[ "${exit_code}" -eq 0 ]] || exit "${exit_code}"
}

run_training() {
  local stage_name="$1" config_path="$2" stage_dir="$3" log_file="$4" model_name="$5"
  ensure_empty "${stage_dir}"
  mkdir -p "${stage_dir}"
  cp "${config_path}" "${stage_dir}/配置快照.yaml"
  python tools/检查V20模型结构.py --config "${config_path}" \
    2>&1 | tee "${stage_dir}/模型结构检查.txt"
  local start_ts
  start_ts="$(date +%s)"
  {
    echo "stage=${stage_name}"
    echo "config=${config_path}"
    echo "start=$(date '+%F %T')"
    echo "source_checkpoint=${SOURCE_CHECKPOINT}"
    echo "source_sha256=$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
    echo "python=$(command -v python)"
  } > "${stage_dir}/运行信息.txt"
  set +e
  python tools/训练与评估_无蒸馏QECR.py --config "${config_path}" \
    2>&1 | tee "${log_file}"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  local elapsed="$(( $(date +%s) - start_ts ))"
  {
    echo "end=$(date '+%F %T')"
    echo "exit_code=${exit_code}"
    echo "elapsed_seconds=${elapsed}"
    printf 'elapsed_hms=%02d:%02d:%02d\n' \
      "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
  } | tee -a "${stage_dir}/运行信息.txt" "${log_file}"
  [[ "${exit_code}" -eq 0 ]] || exit "${exit_code}"
  local trained_checkpoint="${stage_dir}/${model_name}/checkpoint_best.pth"
  [[ -f "${trained_checkpoint}" ]] || {
    echo "错误：训练结束但没有找到最佳权重：${trained_checkpoint}" >&2
    exit 6
  }
  python tools/审计V20训练参数.py \
    --config "${config_path}" \
    --source "${SOURCE_CHECKPOINT}" \
    --trained "${trained_checkpoint}" \
    2>&1 | tee "${stage_dir}/训练后参数审计.txt"
}

summarize() {
  for log_file in "${CONTROL_LOG}" "${QUERY_LOG}" "${GEOMETRY_LOG}"; do
    [[ -f "${log_file}" ]] || {
      echo "日志尚未齐全，不能汇总：${log_file}" >&2
      exit 5
    }
  done
  python tools/汇总V20立体几何深度残差.py \
    --control_log "${CONTROL_LOG}" \
    --query_log "${QUERY_LOG}" \
    --geometry_log "${GEOMETRY_LOG}" \
    --output "${REPORT}"
}

if [[ "${MODE}" != "summary" ]]; then
  preflight
fi
if [[ "${MODE}" == "smoke" ]]; then
  SMOKE_IDS="/root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train_v20_smoke.txt"
  if [[ ! -f "${SMOKE_IDS}" ]]; then
    head -n 24 /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train.txt > "${SMOKE_IDS}"
  fi
  run_training "V20B_24样本冒烟" "${SMOKE_CONFIG}" "${SMOKE_DIR}" "${SMOKE_LOG}" "${SMOKE_MODEL}"
fi
if [[ "${MODE}" == "pair" || "${MODE}" == "control" ]]; then
  run_control
fi
if [[ "${MODE}" == "pair" || "${MODE}" == "query" ]]; then
  run_training "V20A_查询级深度残差" "${QUERY_CONFIG}" "${QUERY_DIR}" "${QUERY_LOG}" "${QUERY_MODEL}"
fi
if [[ "${MODE}" == "pair" || "${MODE}" == "geometry" ]]; then
  run_training "V20B_立体几何深度残差" "${GEOMETRY_CONFIG}" "${GEOMETRY_DIR}" "${GEOMETRY_LOG}" "${GEOMETRY_MODEL}"
fi
if [[ "${MODE}" == "pair" || "${MODE}" == "summary" ]]; then
  summarize
fi
