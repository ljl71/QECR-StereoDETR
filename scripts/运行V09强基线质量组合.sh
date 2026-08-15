#!/usr/bin/env bash
set -euo pipefail

# R0：精确复评V08O最佳checkpoint。
# R1：以同一个V08O权重为冻结检测器，只训练V06B点式3D质量头。
# 默认按R0 -> R1 -> 自动汇总的顺序运行，所有产物写入独立目录。

MODE="${1:-all}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

R0_CONFIG="versions/V08O_几何对齐短程对照/config.yaml"
R1_CONFIG="versions/V09_V08O加点式三维质量排序/config.yaml"
V08O_CHECKPOINT="${PROJECT_ROOT}/outputs/第二创新点_V08空间投影对齐/V08O_公平控制组/qecr_v08o_geometry_control/checkpoint_best.pth"

RUN_ROOT="outputs/最终组合_V09"
R0_DIR="${RUN_ROOT}/R0_V08O直接复评"
R1_DIR="${RUN_ROOT}/V09_V08O加点式三维质量排序"
R0_LOG="${R0_DIR}/控制台.log"
R1_LOG="${R1_DIR}/控制台.log"
REPORT="${RUN_ROOT}/V09结果对照.md"

case "${MODE}" in
  all|r0|r1|summary)
    ;;
  *)
    echo "用法：bash scripts/运行V09强基线质量组合.sh [all|r0|r1|summary]" >&2
    exit 2
    ;;
esac

if [[ "${CONDA_DEFAULT_ENV:-}" != "stereodetr-open" ]]; then
  echo "错误：当前没有激活 stereodetr-open 环境。" >&2
  echo "请先执行：source /root/miniconda3/bin/activate stereodetr-open" >&2
  exit 2
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NUMBA_CUDA_USE_NVIDIA_BINDING="${NUMBA_CUDA_USE_NVIDIA_BINDING:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${V08O_CHECKPOINT}" ]]; then
  echo "错误：没有找到V08O最佳权重：" >&2
  echo "${V08O_CHECKPOINT}" >&2
  echo "请确认PyCharm同步时没有删除服务器outputs目录。" >&2
  exit 4
fi

run_stage() {
  local stage_name="$1"
  local config_path="$2"
  local stage_dir="$3"
  local log_file="$4"
  local evaluate_only="$5"
  local info_file="${stage_dir}/运行信息.txt"
  local snapshot_file="${stage_dir}/配置快照.yaml"

  if [[ -d "${stage_dir}" ]] && [[ -n "$(find "${stage_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "错误：${stage_dir} 已包含实验产物。为避免混合两次运行，本次未启动。" >&2
    exit 3
  fi
  mkdir -p "${stage_dir}"
  cp "${config_path}" "${snapshot_file}"

  local start_ts
  local start_text
  start_ts="$(date +%s)"
  start_text="$(date '+%F %T')"
  {
    echo "stage=${stage_name}"
    echo "config=${config_path}"
    echo "start=${start_text}"
    echo "project_root=${PROJECT_ROOT}"
    echo "python=$(command -v python)"
    echo "v08o_checkpoint=${V08O_CHECKPOINT}"
    echo "v08o_sha256=$(sha256sum "${V08O_CHECKPOINT}" | awk '{print $1}')"
    python --version
    python - <<'PY'
import cv2
import torch

print("opencv={}".format(cv2.__version__))
print("pytorch={}".format(torch.__version__))
print("cuda_runtime={}".format(torch.version.cuda))
print("gpu={}".format(torch.cuda.get_device_name(0)))
PY
  } > "${info_file}"

  set +e
  if [[ "${evaluate_only}" == "true" ]]; then
    python tools/训练与评估_无蒸馏QECR.py \
      --evaluate_only \
      --config "${config_path}" \
      2>&1 | tee "${log_file}"
  else
    python tools/训练与评估_无蒸馏QECR.py \
      --config "${config_path}" \
      2>&1 | tee "${log_file}"
  fi
  local exit_code="${PIPESTATUS[0]}"
  set -e

  local end_ts
  local elapsed
  end_ts="$(date +%s)"
  elapsed="$((end_ts - start_ts))"
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
  python tools/检查QECR消融配置.py
  python tools/测试Checkpoint兼容白名单.py
  python tools/测试3D质量排序.py
fi

if [[ "${MODE}" == "all" || "${MODE}" == "r0" ]]; then
  run_stage "R0_V08O直接复评" "${R0_CONFIG}" "${R0_DIR}" "${R0_LOG}" true
fi

if [[ "${MODE}" == "all" || "${MODE}" == "r1" ]]; then
  run_stage "R1_V08O加V06B" "${R1_CONFIG}" "${R1_DIR}" "${R1_LOG}" false
fi

if [[ -f "${R0_LOG}" && -f "${R1_LOG}" ]]; then
  python tools/汇总V09强基线质量组合.py \
    --r0_log "${R0_LOG}" \
    --r1_log "${R1_LOG}" \
    --output "${REPORT}"
else
  echo "当前只完成部分阶段；R0和R1日志都存在后再自动生成结果对照。"
  echo "可执行：bash scripts/运行V09强基线质量组合.sh summary"
fi
