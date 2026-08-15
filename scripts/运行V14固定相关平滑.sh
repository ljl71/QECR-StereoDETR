#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pair}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SOURCE_CHECKPOINT="${PROJECT_ROOT}/outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
RUN_ROOT="outputs/固定相关平滑_V14/零训练复评"
CONTROL_CONFIG="versions/V14O_零训练原始相关复评/config.yaml"
CANDIDATE_CONFIG="versions/V14S_零训练s4固定平滑/config.yaml"
CONTROL_DIR="${RUN_ROOT}/V14O_原始相关"
CANDIDATE_DIR="${RUN_ROOT}/V14S_s4固定平滑"
CONTROL_MODEL="qecr_v14o_zero_train_original_correlation"
CANDIDATE_MODEL="qecr_v14s_zero_train_s4_fixed_smoothing"
CONTROL_LOG="${CONTROL_DIR}/控制台.log"
CANDIDATE_LOG="${CANDIDATE_DIR}/控制台.log"
REPORT="${RUN_ROOT}/V14零训练复评结果.md"

case "${MODE}" in
  pair|control|smooth|summary)
    ;;
  *)
    echo "用法：bash scripts/运行V14固定相关平滑.sh [pair|control|smooth|summary]" >&2
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
  python tools/测试V14固定相关平滑.py
  python tools/汇总V14固定相关平滑.py --self_test
}

run_evaluation() {
  local stage_name="$1"
  local config_path="$2"
  local stage_dir="$3"
  local model_name="$4"
  local log_file="$5"
  local checkpoint_dir="${stage_dir}/${model_name}"
  local checkpoint_path="${checkpoint_dir}/checkpoint_best.pth"
  local info_file="${stage_dir}/运行信息.txt"

  if [[ -d "${stage_dir}" ]] && [[ -n "$(find "${stage_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "错误：${stage_dir}已经包含实验产物；为避免覆盖，本次未启动。" >&2
    exit 3
  fi
  mkdir -p "${checkpoint_dir}"
  cp "${config_path}" "${stage_dir}/配置快照.yaml"
  cp --reflink=auto "${SOURCE_CHECKPOINT}" "${checkpoint_path}"

  local source_sha
  local copied_sha
  source_sha="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
  copied_sha="$(sha256sum "${checkpoint_path}" | awk '{print $1}')"
  if [[ "${source_sha}" != "${copied_sha}" ]]; then
    echo "错误：复制后的checkpoint校验值不一致。" >&2
    exit 5
  fi

  {
    echo "stage=${stage_name}"
    echo "config=${config_path}"
    echo "start=$(date '+%F %T')"
    echo "training=none"
    echo "source_checkpoint=${SOURCE_CHECKPOINT}"
    echo "source_sha256=${source_sha}"
    echo "copied_checkpoint=${checkpoint_path}"
    echo "copied_sha256=${copied_sha}"
    echo "python=$(command -v python)"
    python --version
    python - <<'PY'
import torch
print("pytorch={}".format(torch.__version__))
print("cuda_runtime={}".format(torch.version.cuda))
print("gpu={}".format(torch.cuda.get_device_name(0)))
PY
  } > "${info_file}"

  local start_ts
  start_ts="$(date +%s)"
  set +e
  python tools/训练与评估_无蒸馏QECR.py \
    --evaluate_only \
    --config "${config_path}" \
    2>&1 | tee "${log_file}"
  local exit_code="${PIPESTATUS[0]}"
  set -e

  local elapsed="$(( $(date +%s) - start_ts ))"
  {
    echo "end=$(date '+%F %T')"
    echo "exit_code=${exit_code}"
    echo "elapsed_seconds=${elapsed}"
    printf 'elapsed_hms=%02d:%02d:%02d\n' \
      "$((elapsed / 3600))" \
      "$(((elapsed % 3600) / 60))" \
      "$((elapsed % 60))"
  } | tee -a "${info_file}" "${log_file}"

  if [[ "${exit_code}" -ne 0 ]]; then
    echo "${stage_name}失败，退出码=${exit_code}" >&2
    exit "${exit_code}"
  fi
}

summarize() {
  if [[ ! -f "${CONTROL_LOG}" || ! -f "${CANDIDATE_LOG}" ]]; then
    echo "V14O和V14S日志尚未齐全，不能汇总。" >&2
    exit 5
  fi
  python tools/汇总V14固定相关平滑.py \
    --control_log "${CONTROL_LOG}" \
    --candidate_log "${CANDIDATE_LOG}" \
    --output "${REPORT}"
}

if [[ "${MODE}" != "summary" ]]; then
  preflight
fi

if [[ "${MODE}" == "pair" || "${MODE}" == "control" ]]; then
  run_evaluation \
    "V14O_零训练原始相关复评" \
    "${CONTROL_CONFIG}" "${CONTROL_DIR}" "${CONTROL_MODEL}" "${CONTROL_LOG}"
fi

if [[ "${MODE}" == "pair" || "${MODE}" == "smooth" ]]; then
  run_evaluation \
    "V14S_零训练s4固定平滑" \
    "${CANDIDATE_CONFIG}" "${CANDIDATE_DIR}" "${CANDIDATE_MODEL}" "${CANDIDATE_LOG}"
fi

if [[ "${MODE}" == "pair" || "${MODE}" == "summary" ]]; then
  summarize
fi
