#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pair}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SOURCE_CHECKPOINT="${PROJECT_ROOT}/outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
RUN_ROOT="outputs/第二创新点_V11动态深度上采样"
CONTROL_CONFIG="versions/V11O_双线性深度上采样控制/config.yaml"
STAGE1_CONFIG="versions/V11A_第一级动态深度上采样/config.yaml"
STAGE2_CONFIG="versions/V11B_两级动态深度上采样/config.yaml"
SMOKE_CONFIG="configs/AutoDL_V11A_动态上采样冒烟.yaml"
CONTROL_DIR="${RUN_ROOT}/V11O_双线性训练控制"
STAGE1_DIR="${RUN_ROOT}/V11A_第一级动态上采样"
STAGE2_DIR="${RUN_ROOT}/V11B_两级动态上采样"
SMOKE_DIR="${RUN_ROOT}/冒烟测试"
CONTROL_LOG="${CONTROL_DIR}/控制台.log"
STAGE1_LOG="${STAGE1_DIR}/控制台.log"
STAGE2_LOG="${STAGE2_DIR}/控制台.log"
SMOKE_LOG="${SMOKE_DIR}/控制台.log"
REPORT="${RUN_ROOT}/V11配对结果.md"

case "${MODE}" in
  smoke|pair|control|stage1|stage2|summary)
    ;;
  *)
    echo "用法：bash scripts/运行V11动态深度上采样.sh [smoke|pair|control|stage1|stage2|summary]" >&2
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
  local log_file="$4"
  local info_file="${stage_dir}/运行信息.txt"
  local structure_file="${stage_dir}/模型结构检查.txt"

  if [[ -d "${stage_dir}" ]] && [[ -n "$(find "${stage_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "错误：${stage_dir}已经包含实验产物；为避免覆盖，本次未启动。" >&2
    exit 3
  fi
  mkdir -p "${stage_dir}"
  cp "${config_path}" "${stage_dir}/配置快照.yaml"

  python tools/检查V11模型结构.py --config "${config_path}" \
    2>&1 | tee "${structure_file}"

  local start_ts
  start_ts="$(date +%s)"
  {
    echo "stage=${stage_name}"
    echo "config=${config_path}"
    echo "start=$(date '+%F %T')"
    echo "source_checkpoint=${SOURCE_CHECKPOINT}"
    echo "source_sha256=$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
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
  if [[ ! -f "${CONTROL_LOG}" || ! -f "${STAGE1_LOG}" ]]; then
    echo "V11O和V11A日志尚未齐全，不能汇总。" >&2
    return 5
  fi
  local stage2_args=()
  if [[ -f "${STAGE2_LOG}" ]]; then
    stage2_args=(--stage2_log "${STAGE2_LOG}")
  fi
  python tools/汇总V11动态深度上采样.py \
    --control_log "${CONTROL_LOG}" \
    --stage1_log "${STAGE1_LOG}" \
    "${stage2_args[@]}" \
    --output "${REPORT}"
}

if [[ "${MODE}" != "summary" ]]; then
  python tools/检查QECR消融配置.py
  python tools/测试V11动态深度上采样.py
  python tools/汇总V11动态深度上采样.py --self_test
fi

if [[ "${MODE}" == "smoke" ]]; then
  SMOKE_IDS="/root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train_v11_smoke.txt"
  if [[ ! -f "${SMOKE_IDS}" ]]; then
    head -n 24 /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train.txt > "${SMOKE_IDS}"
  fi
  run_stage "V11A_24样本冒烟" "${SMOKE_CONFIG}" "${SMOKE_DIR}" "${SMOKE_LOG}"
fi

if [[ "${MODE}" == "pair" || "${MODE}" == "control" ]]; then
  run_stage "V11O_双线性训练控制" "${CONTROL_CONFIG}" "${CONTROL_DIR}" "${CONTROL_LOG}"
fi

if [[ "${MODE}" == "pair" || "${MODE}" == "stage1" ]]; then
  run_stage "V11A_第一级动态上采样" "${STAGE1_CONFIG}" "${STAGE1_DIR}" "${STAGE1_LOG}"
fi

if [[ "${MODE}" == "pair" || "${MODE}" == "summary" ]]; then
  summarize
fi

if [[ "${MODE}" == "stage2" ]]; then
  if [[ ! -f "${REPORT}" ]]; then
    summarize
  fi
  if ! grep -q 'V11A自动判定：\*\*精度通过\*\*' "${REPORT}"; then
    echo "V11A没有达到精度通过门槛，V11B未启动。" >&2
    exit 6
  fi
  run_stage "V11B_两级动态上采样" "${STAGE2_CONFIG}" "${STAGE2_DIR}" "${STAGE2_LOG}"
  summarize
fi
