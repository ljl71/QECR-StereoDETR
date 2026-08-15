#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="versions/V09_V08O加点式三维质量排序/config.yaml"
CHECKPOINT="outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
VELODYNE_ZIP="/autodl-pub/data/KITTI_Object/raw/data_object_velodyne.zip"
RUN_ROOT="outputs/第二创新点_V12分组相关诊断"

case "${MODE}" in
  selftest)
    python tools/诊断分组相关信息_V12G0.py --self_test
    exit 0
    ;;
  smoke)
    NUM_SAMPLES=20
    POINTS_PER_REGION=128
    OUTPUT_DIR="${RUN_ROOT}/G0_20样本冒烟"
    ;;
  formal)
    NUM_SAMPLES=100
    POINTS_PER_REGION=384
    OUTPUT_DIR="${RUN_ROOT}/G0_100样本正式诊断"
    ;;
  *)
    echo "用法：bash scripts/运行V12分组相关诊断.sh [selftest|smoke|formal]" >&2
    exit 2
    ;;
esac

if [[ "${CONDA_DEFAULT_ENV:-}" != "stereodetr-open" ]]; then
  echo "错误：请先激活 stereodetr-open 环境。" >&2
  exit 2
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NUMBA_CUDA_USE_NVIDIA_BINDING="${NUMBA_CUDA_USE_NVIDIA_BINDING:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

for required in "${CONFIG}" "${CHECKPOINT}" "${VELODYNE_ZIP}"; do
  if [[ ! -f "${required}" ]]; then
    echo "错误：缺少输入 ${required}" >&2
    exit 4
  fi
done

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "错误：${OUTPUT_DIR} 已有产物；为避免覆盖，本次未启动。" >&2
  exit 3
fi

mkdir -p "${OUTPUT_DIR}"
cp "${CONFIG}" "${OUTPUT_DIR}/V09配置快照.yaml"
python tools/诊断分组相关信息_V12G0.py --self_test

START_TS="$(date +%s)"
{
  echo "mode=${MODE}"
  echo "start=$(date '+%F %T')"
  echo "config=${CONFIG}"
  echo "checkpoint=${CHECKPOINT}"
  echo "checkpoint_sha256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
  echo "velodyne_zip=${VELODYNE_ZIP}"
  echo "num_samples=${NUM_SAMPLES}"
  echo "points_per_region=${POINTS_PER_REGION}"
  echo "groups=4,8,16"
  echo "scales=4,8,16"
  echo "python=$(command -v python)"
  python --version
  python - <<'PY'
import torch
print("pytorch={}".format(torch.__version__))
print("cuda_runtime={}".format(torch.version.cuda))
print("gpu={}".format(torch.cuda.get_device_name(0)))
PY
} > "${OUTPUT_DIR}/运行信息.txt"

set +e
python tools/诊断分组相关信息_V12G0.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --velodyne_zip "${VELODYNE_ZIP}" \
  --num_samples "${NUM_SAMPLES}" \
  --points_per_region "${POINTS_PER_REGION}" \
  --groups 4,8,16 \
  --scales 4,8,16 \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/控制台.log"
EXIT_CODE="${PIPESTATUS[0]}"
set -e

ELAPSED="$(( $(date +%s) - START_TS ))"
{
  echo "end=$(date '+%F %T')"
  echo "exit_code=${EXIT_CODE}"
  echo "elapsed_seconds=${ELAPSED}"
  printf 'elapsed_hms=%02d:%02d:%02d\n' \
    "$((ELAPSED / 3600))" \
    "$(((ELAPSED % 3600) / 60))" \
    "$((ELAPSED % 60))"
} | tee -a "${OUTPUT_DIR}/运行信息.txt" "${OUTPUT_DIR}/控制台.log"

if [[ "${EXIT_CODE}" -ne 0 ]]; then
  exit "${EXIT_CODE}"
fi

echo
cat "${OUTPUT_DIR}/诊断报告.md"

