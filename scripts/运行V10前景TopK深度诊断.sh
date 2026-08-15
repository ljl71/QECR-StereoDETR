#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SOURCE_CHECKPOINT="${PROJECT_ROOT}/outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
RUN_ROOT="outputs/第二创新点_V10前景TopK深度读出诊断"
CONTROL_CONFIG="versions/V10O_全分布深度读出控制/config.yaml"
TOPK2_CONFIG="versions/V10A_前景TopK2深度读出/config.yaml"
TOPK4_CONFIG="versions/V10B_前景TopK4深度读出/config.yaml"
CONTROL_DIR="${RUN_ROOT}/V10O_全分布控制"
TOPK2_DIR="${RUN_ROOT}/V10A_前景TopK2"
TOPK4_DIR="${RUN_ROOT}/V10B_前景TopK4"
CONTROL_MODEL="qecr_v10o_full_distribution_control"
TOPK2_MODEL="qecr_v10a_foreground_topk2"
TOPK4_MODEL="qecr_v10b_foreground_topk4"
CONTROL_LOG="${CONTROL_DIR}/控制台.log"
TOPK2_LOG="${TOPK2_DIR}/控制台.log"
TOPK4_LOG="${TOPK4_DIR}/控制台.log"
REPORT="${RUN_ROOT}/V10诊断结果.md"

case "${MODE}" in
  all|control|k2|k4|summary)
    ;;
  *)
    echo "用法：bash scripts/运行V10前景TopK深度诊断.sh [all|control|k2|k4|summary]" >&2
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

run_stage() {
  local stage_name="$1"
  local config_path="$2"
  local stage_dir="$3"
  local model_name="$4"
  local log_file="$5"
  local model_dir="${stage_dir}/${model_name}"
  local info_file="${stage_dir}/运行信息.txt"

  if [[ -d "${stage_dir}" ]] && [[ -n "$(find "${stage_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "错误：${stage_dir}已经包含实验产物；为避免混合运行，本次未覆盖。" >&2
    exit 3
  fi
  mkdir -p "${model_dir}"
  cp "${config_path}" "${stage_dir}/配置快照.yaml"
  cp "${SOURCE_CHECKPOINT}" "${model_dir}/checkpoint_best.pth"

  local start_ts
  start_ts="$(date +%s)"
  {
    echo "stage=${stage_name}"
    echo "config=${config_path}"
    echo "start=$(date '+%F %T')"
    echo "source_checkpoint=${SOURCE_CHECKPOINT}"
    echo "source_sha256=$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
    echo "copied_sha256=$(sha256sum "${model_dir}/checkpoint_best.pth" | awk '{print $1}')"
    echo "python=$(command -v python)"
    python --version
    python - <<'PY'
import torch
print("pytorch={}".format(torch.__version__))
print("cuda_runtime={}".format(torch.version.cuda))
print("gpu={}".format(torch.cuda.get_device_name(0)))
PY
  } > "${info_file}"

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

if [[ "${MODE}" != "summary" ]]; then
  python tools/测试前景TopK深度读出.py
  python tools/汇总V10前景TopK深度诊断.py --self_test
fi

if [[ "${MODE}" == "all" || "${MODE}" == "control" ]]; then
  run_stage "V10O_原始全分布控制" "${CONTROL_CONFIG}" "${CONTROL_DIR}" "${CONTROL_MODEL}" "${CONTROL_LOG}"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "k2" ]]; then
  run_stage "V10A_前景TopK2" "${TOPK2_CONFIG}" "${TOPK2_DIR}" "${TOPK2_MODEL}" "${TOPK2_LOG}"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "k4" ]]; then
  run_stage "V10B_前景TopK4" "${TOPK4_CONFIG}" "${TOPK4_DIR}" "${TOPK4_MODEL}" "${TOPK4_LOG}"
fi

if [[ -f "${CONTROL_LOG}" && -f "${TOPK2_LOG}" && -f "${TOPK4_LOG}" ]]; then
  python tools/汇总V10前景TopK深度诊断.py \
    --control_log "${CONTROL_LOG}" \
    --topk2_log "${TOPK2_LOG}" \
    --topk4_log "${TOPK4_LOG}" \
    --output "${REPORT}"
else
  echo "当前只有部分阶段完成；三份日志齐全后再生成结果报告。"
fi

