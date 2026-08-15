#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pair}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SOURCE_CHECKPOINT="${PROJECT_ROOT}/outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
RUN_ROOT="outputs/第二创新点_V12分组相关门控"
CONTROL_CONFIG="versions/V12O_原始相关公平控制/config.yaml"
CANDIDATE_CONFIG="versions/V12A_s4十六组轻量门控/config.yaml"
SMOKE_CONFIG="configs/AutoDL_V12A_分组相关门控冒烟.yaml"
CONTROL_DIR="${RUN_ROOT}/V12O_原始相关公平控制"
CANDIDATE_DIR="${RUN_ROOT}/V12A_s4十六组轻量门控"
SMOKE_DIR="${RUN_ROOT}/冒烟测试"
CONTROL_LOG="${CONTROL_DIR}/控制台.log"
CANDIDATE_LOG="${CANDIDATE_DIR}/控制台.log"
SMOKE_LOG="${SMOKE_DIR}/控制台.log"
REPORT="${RUN_ROOT}/V12配对结果.md"
CONTROL_CKPT="${CONTROL_DIR}/qecr_v12o_original_correlation_control/checkpoint_best.pth"
CANDIDATE_CKPT="${CANDIDATE_DIR}/qecr_v12a_s4_group16_light_gate/checkpoint_best.pth"
SECOND_ROOT="${RUN_ROOT}/第二随机种子_445"
SECOND_CONTROL_CONFIG="versions/V12O_种子445原始相关控制/config.yaml"
SECOND_CANDIDATE_CONFIG="versions/V12A_种子445s4十六组轻量门控/config.yaml"
SECOND_CONTROL_DIR="${SECOND_ROOT}/V12O_原始相关控制"
SECOND_CANDIDATE_DIR="${SECOND_ROOT}/V12A_s4十六组轻量门控"
SECOND_CONTROL_LOG="${SECOND_CONTROL_DIR}/控制台.log"
SECOND_CANDIDATE_LOG="${SECOND_CANDIDATE_DIR}/控制台.log"
SECOND_REPORT="${RUN_ROOT}/V12双随机种子复核报告.md"
LATENCY_DIR="${RUN_ROOT}/正式时延测试"
if [[ "${MODE}" == "latency_optimized" ]]; then
  LATENCY_DIR="${RUN_ROOT}/正式时延复测_等价优化"
fi

case "${MODE}" in
  smoke|pair|control|candidate|summary|latency|latency_optimized|seed445_pair|seed445_summary)
    ;;
  *)
    echo "用法：bash scripts/运行V12轻量分组相关.sh [smoke|pair|control|candidate|summary|latency|latency_optimized|seed445_pair|seed445_summary]" >&2
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
  python tools/检查V12模型结构.py --config "${config_path}" \
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

  local model_name
  model_name="$(python - "${config_path}" <<'PY'
import sys
from lib.helpers.config_helper import load_config
print(load_config(sys.argv[1])["model_name"])
PY
)"
  local trained_checkpoint="${stage_dir}/${model_name}/checkpoint_best.pth"
  if [[ ! -f "${trained_checkpoint}" ]]; then
    echo "${stage_name}没有生成checkpoint_best.pth" >&2
    exit 7
  fi
  python tools/审计V12训练参数.py \
    --config "${config_path}" \
    --source "${SOURCE_CHECKPOINT}" \
    --trained "${trained_checkpoint}" \
    2>&1 | tee "${stage_dir}/训练后参数审计.txt"
}

summarize() {
  if [[ ! -f "${CONTROL_LOG}" || ! -f "${CANDIDATE_LOG}" ]]; then
    echo "V12O和V12A日志尚未齐全，不能汇总。" >&2
    return 5
  fi
  python tools/汇总V12分组相关门控.py \
    --control_log "${CONTROL_LOG}" \
    --candidate_log "${CANDIDATE_LOG}" \
    --output "${REPORT}"
}

summarize_second_seed() {
  for required_log in \
    "${CONTROL_LOG}" "${CANDIDATE_LOG}" \
    "${SECOND_CONTROL_LOG}" "${SECOND_CANDIDATE_LOG}"
  do
    if [[ ! -f "${required_log}" ]]; then
      echo "缺少双种子汇总日志：${required_log}" >&2
      return 5
    fi
  done
  python tools/汇总V12双种子复核.py \
    --control_444_log "${CONTROL_LOG}" \
    --candidate_444_log "${CANDIDATE_LOG}" \
    --control_445_log "${SECOND_CONTROL_LOG}" \
    --candidate_445_log "${SECOND_CANDIDATE_LOG}" \
    --output "${SECOND_REPORT}"
}

preflight() {
  python tools/检查QECR消融配置.py
  python tools/测试V12轻量分组相关.py
  python tools/汇总V12分组相关门控.py --self_test
  python tools/汇总V12双种子复核.py --self_test
}

run_latency() {
  if [[ ! -f "${REPORT}" ]]; then
    summarize
  fi
  if ! grep -q 'V12A自动判定：\*\*精度通过\*\*' "${REPORT}"; then
    echo "V12A尚未达到精度通过门槛，不启动正式时延测试。" >&2
    exit 6
  fi
  if [[ ! -f "${CONTROL_CKPT}" || ! -f "${CANDIDATE_CKPT}" ]]; then
    echo "V12O或V12A最佳checkpoint缺失，不能测速。" >&2
    exit 5
  fi
  if [[ -d "${LATENCY_DIR}" ]] && [[ -n "$(find "${LATENCY_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "错误：${LATENCY_DIR}已有结果；为避免覆盖，本次未启动。" >&2
    exit 3
  fi
  mkdir -p "${LATENCY_DIR}"

  run_one_latency() {
    local label="$1"
    local config="$2"
    local checkpoint="$3"
    python tools/测量QECR端到端时延.py \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --warmup 20 \
      --steps 100 \
      2>&1 | tee "${LATENCY_DIR}/${label}.log"
  }

  # Alternate order to reduce temperature/clock-order bias.
  run_one_latency "第1组_V12O" "${CONTROL_CONFIG}" "${CONTROL_CKPT}"
  run_one_latency "第1组_V12A" "${CANDIDATE_CONFIG}" "${CANDIDATE_CKPT}"
  run_one_latency "第2组_V12A" "${CANDIDATE_CONFIG}" "${CANDIDATE_CKPT}"
  run_one_latency "第2组_V12O" "${CONTROL_CONFIG}" "${CONTROL_CKPT}"
  run_one_latency "第3组_V12O" "${CONTROL_CONFIG}" "${CONTROL_CKPT}"
  run_one_latency "第3组_V12A" "${CANDIDATE_CONFIG}" "${CANDIDATE_CKPT}"
  grep -HE 'GPU:|parameters:|median_ms:|mean_ms:|p90_ms:|fps_from_median:' \
    "${LATENCY_DIR}"/*.log | tee "${LATENCY_DIR}/汇总.txt"
  python tools/汇总V12时延.py \
    --latency_dir "${LATENCY_DIR}" \
    --output "${LATENCY_DIR}/时延判定.md"
}

if [[ "${MODE}" != "summary" && "${MODE}" != "seed445_summary" && "${MODE}" != "latency" && "${MODE}" != "latency_optimized" ]]; then
  preflight
fi

if [[ "${MODE}" == "smoke" ]]; then
  SMOKE_IDS="/root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train_v12_smoke.txt"
  if [[ ! -f "${SMOKE_IDS}" ]]; then
    head -n 24 /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train.txt > "${SMOKE_IDS}"
  fi
  run_stage "V12A_24样本冒烟" "${SMOKE_CONFIG}" "${SMOKE_DIR}" "${SMOKE_LOG}"
fi

if [[ "${MODE}" == "pair" || "${MODE}" == "control" ]]; then
  run_stage "V12O_原始相关公平控制" "${CONTROL_CONFIG}" "${CONTROL_DIR}" "${CONTROL_LOG}"
fi

if [[ "${MODE}" == "pair" || "${MODE}" == "candidate" ]]; then
  run_stage "V12A_s4十六组轻量门控" "${CANDIDATE_CONFIG}" "${CANDIDATE_DIR}" "${CANDIDATE_LOG}"
fi

if [[ "${MODE}" == "pair" || "${MODE}" == "summary" ]]; then
  summarize
fi

if [[ "${MODE}" == "seed445_pair" ]]; then
  run_stage "V12O_种子445原始相关控制" \
    "${SECOND_CONTROL_CONFIG}" "${SECOND_CONTROL_DIR}" "${SECOND_CONTROL_LOG}"
  run_stage "V12A_种子445s4十六组轻量门控" \
    "${SECOND_CANDIDATE_CONFIG}" "${SECOND_CANDIDATE_DIR}" "${SECOND_CANDIDATE_LOG}"
  summarize_second_seed
fi

if [[ "${MODE}" == "seed445_summary" ]]; then
  summarize_second_seed
fi

if [[ "${MODE}" == "latency" || "${MODE}" == "latency_optimized" ]]; then
  run_latency
fi
