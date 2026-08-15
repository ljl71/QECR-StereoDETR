#!/usr/bin/env bash
set -euo pipefail

# 统一运行 V08O/A/B/C，并把日志、配置快照、环境信息和 checkpoint
# 收纳在同一个版本目录中。为避免误覆盖，已有控制台日志时直接退出。

VERSION="${1:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

case "${VERSION}" in
  V08O)
    CONFIG="versions/V08O_几何对齐短程对照/config.yaml"
    RUN_DIR="outputs/第二创新点_V08空间投影对齐/V08O_公平控制组"
    ;;
  V08A)
    CONFIG="versions/V08A_分阶段三维角点对齐/config.yaml"
    RUN_DIR="outputs/第二创新点_V08空间投影对齐/V08A_三维角点对齐"
    ;;
  V08B)
    CONFIG="versions/V08B_分阶段左目投影对齐/config.yaml"
    RUN_DIR="outputs/第二创新点_V08空间投影对齐/V08B_左目投影对齐"
    ;;
  V08C)
    CONFIG="versions/V08C_空间投影协同对齐/config.yaml"
    RUN_DIR="outputs/第二创新点_V08空间投影对齐/V08C_空间投影协同对齐"
    ;;
  *)
    echo "用法：bash scripts/运行V08短程实验.sh V08O|V08A|V08B|V08C" >&2
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

mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/控制台.log"
INFO_FILE="${RUN_DIR}/运行信息.txt"
CONFIG_SNAPSHOT="${RUN_DIR}/配置快照.yaml"

if [[ -e "${LOG_FILE}" ]]; then
  echo "错误：${LOG_FILE} 已存在，为避免覆盖，本次未启动。" >&2
  exit 3
fi

cp "${CONFIG}" "${CONFIG_SNAPSHOT}"
START_TS="$(date +%s)"
START_TEXT="$(date '+%F %T')"

{
  echo "version=${VERSION}"
  echo "config=${CONFIG}"
  echo "start=${START_TEXT}"
  echo "project_root=${PROJECT_ROOT}"
  echo "python=$(command -v python)"
  python --version
  python - <<'PY'
import cv2
import torch

print("opencv={}".format(cv2.__version__))
print("pytorch={}".format(torch.__version__))
print("cuda_runtime={}".format(torch.version.cuda))
print("gpu={}".format(torch.cuda.get_device_name(0)))
PY
} > "${INFO_FILE}"

set +e
python tools/训练与评估_无蒸馏QECR.py \
  --config "${CONFIG}" \
  2>&1 | tee "${LOG_FILE}"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e

END_TS="$(date +%s)"
END_TEXT="$(date '+%F %T')"
ELAPSED="$((END_TS - START_TS))"

{
  echo "end=${END_TEXT}"
  echo "exit_code=${TRAIN_EXIT}"
  echo "elapsed_seconds=${ELAPSED}"
  printf 'elapsed_hms=%02d:%02d:%02d\n' \
    "$((ELAPSED / 3600))" \
    "$(((ELAPSED % 3600) / 60))" \
    "$((ELAPSED % 60))"
} | tee -a "${INFO_FILE}" "${LOG_FILE}"

exit "${TRAIN_EXIT}"
