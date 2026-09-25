#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-status}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_ROOT="outputs/V23分组保真预聚合"
SOURCE_CHECKPOINT="${PROJECT_ROOT}/outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
IMPORT_CHECKPOINT="${V23_SOURCE_CHECKPOINT:-/root/autodl-tmp/QECR-StereoDETR/outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth}"
EXPECTED_V09_SHA256="6b673fef2a54e2a5fbc44be731b24914770fffebaf1c20a0cacc43228dd316b4"

CONFIG_O="versions/V23O_原始相关公平控制/config.yaml"
CONFIG_A="versions/V23A_残差式视差空间微聚合/config.yaml"
CONFIG_B="versions/V23B_分组保真空间视差预聚合/config.yaml"
CONFIG_O_445="versions/V23O_种子445原始相关公平控制/config.yaml"
CONFIG_A_445="versions/V23A_种子445残差式视差空间微聚合/config.yaml"
CONFIG_B_445="versions/V23B_种子445分组保真空间视差预聚合/config.yaml"
CONFIG_Q="versions/V23Q_GPSD查询级质量重标定/config.yaml"
CONFIG_SMOKE="configs/AutoDL_V23B_GPSD冒烟.yaml"

DIR_O="${RUN_ROOT}/V23O_原始相关公平控制"
DIR_A="${RUN_ROOT}/V23A_残差式视差空间微聚合"
DIR_B="${RUN_ROOT}/V23B_分组保真空间视差预聚合"
DIR_Q="${RUN_ROOT}/V23Q_GPSD查询级质量重标定"
DIR_SMOKE="${RUN_ROOT}/冒烟测试"
SECOND_ROOT="${RUN_ROOT}/第二随机种子_445"
DIR_O_445="${SECOND_ROOT}/V23O_原始相关公平控制"
DIR_A_445="${SECOND_ROOT}/V23A_残差式视差空间微聚合"
DIR_B_445="${SECOND_ROOT}/V23B_分组保真空间视差预聚合"

LOG_O="${DIR_O}/控制台.log"
LOG_A="${DIR_A}/控制台.log"
LOG_B="${DIR_B}/控制台.log"
LOG_Q="${DIR_Q}/控制台.log"
LOG_SMOKE="${DIR_SMOKE}/控制台.log"
LOG_O_445="${DIR_O_445}/控制台.log"
LOG_A_445="${DIR_A_445}/控制台.log"
LOG_B_445="${DIR_B_445}/控制台.log"

CKPT_O="${DIR_O}/qecr_v23o_original_correlation_control/checkpoint_best.pth"
CKPT_A="${DIR_A}/qecr_v23a_rdsa/checkpoint_best.pth"
CKPT_B="${DIR_B}/qecr_v23b_gpsd/checkpoint_best.pth"
CKPT_Q="${DIR_Q}/qecr_v23q_gpsd_quality_recalibration/checkpoint_best.pth"

REPORT_PAIR="${RUN_ROOT}/V23配对结果.md"
REPORT_SEEDS="${RUN_ROOT}/V23双随机种子复核报告.md"
REPORT_QUALITY="${RUN_ROOT}/V23最终质量重标定结果.md"
G8_DIR="${RUN_ROOT}/G8_100样本诊断"
G8_REPORT="${G8_DIR}/诊断报告.md"
LATENCY_DIR="${RUN_ROOT}/正式时延测试"

case "${MODE}" in
  preflight|g8|smoke|pair|control|rdsa|gpsd|summary|seed445_pair|seed445_summary|latency|quality|quality_summary|status|all)
    ;;
  *)
    echo "用法：bash scripts/运行V23分组视差预聚合.sh [preflight|g8|smoke|pair|control|rdsa|gpsd|summary|seed445_pair|seed445_summary|latency|quality|quality_summary|status|all]" >&2
    exit 2
    ;;
esac

if [[ "${CONDA_DEFAULT_ENV:-}" != "stereodetr-open" ]] && [[ "${MODE}" != "status" ]]; then
  echo "错误：请先激活stereodetr-open环境。" >&2
  exit 2
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NUMBA_CUDA_USE_NVIDIA_BINDING="${NUMBA_CUDA_USE_NVIDIA_BINDING:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

prepare_source_checkpoint() {
  if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
    if [[ ! -f "${IMPORT_CHECKPOINT}" ]]; then
      echo "错误：原项目内没有V09最佳权重，指定的外部权重也不存在。" >&2
      echo "预期导入路径：${IMPORT_CHECKPOINT}" >&2
      exit 4
    fi
    mkdir -p "$(dirname "${SOURCE_CHECKPOINT}")"
    cp "${IMPORT_CHECKPOINT}" "${SOURCE_CHECKPOINT}"
    echo "已从指定外部路径复制V09权重到原项目。"
  fi
  local actual_sha
  actual_sha="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${EXPECTED_V09_SHA256}" ]]; then
    echo "错误：V09权重SHA256不符合固定实验起点。" >&2
    echo "actual=${actual_sha}" >&2
    echo "expected=${EXPECTED_V09_SHA256}" >&2
    exit 4
  fi
  echo "V09固定起点校验通过：${actual_sha}"
}

preflight() {
  python tools/检查QECR消融配置.py
  python tools/诊断GPSD信息保真_G8.py --self_test
  python tools/测试V23分组视差预聚合.py
  python tools/汇总V23分组视差预聚合.py --self_test
  python tools/汇总V23双随机种子.py --self_test
  python tools/汇总V23时延.py --self_test
  python tools/汇总V23质量重标定.py --self_test
  for config in "${CONFIG_O}" "${CONFIG_A}" "${CONFIG_B}"; do
    python tools/检查V23模型结构.py --config "${config}"
  done
  python -m py_compile \
    lib/models/monodetr/depth_predictor/depth_predictor_lightstereo.py \
    lib/models/monodetr/stereodetr.py \
    tools/诊断GPSD信息保真_G8.py \
    tools/测试V23分组视差预聚合.py \
    tools/检查V23模型结构.py \
    tools/审计V23训练参数.py \
    tools/审计V23质量训练参数.py \
    tools/汇总V23分组视差预聚合.py \
    tools/汇总V23双随机种子.py \
    tools/汇总V23时延.py \
    tools/汇总V23质量重标定.py
  echo "V23训练前检查全部通过"
}

require_empty_dir() {
  local path="$1"
  if [[ -d "${path}" ]] && [[ -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "错误：${path}已有实验产物。为避免覆盖，本次未启动。" >&2
    exit 3
  fi
}

run_stage() {
  local label="$1"
  local config="$2"
  local stage_dir="$3"
  local log="$4"
  local audit_type="${5:-cost}"
  require_empty_dir "${stage_dir}"
  mkdir -p "${stage_dir}"
  cp "${config}" "${stage_dir}/配置快照.yaml"
  if [[ "${audit_type}" == "cost" ]]; then
    python tools/检查V23模型结构.py --config "${config}" \
      2>&1 | tee "${stage_dir}/模型结构检查.txt"
  fi

  local start_ts
  start_ts="$(date +%s)"
  {
    echo "stage=${label}"
    echo "config=${config}"
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
  } > "${stage_dir}/运行信息.txt"

  set +e
  python tools/训练与评估_无蒸馏QECR.py --config "${config}" \
    2>&1 | tee "${log}"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  local elapsed="$(( $(date +%s) - start_ts ))"
  {
    echo "end=$(date '+%F %T')"
    echo "exit_code=${exit_code}"
    echo "elapsed_seconds=${elapsed}"
    printf 'elapsed_hms=%02d:%02d:%02d\n' \
      "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
  } | tee -a "${stage_dir}/运行信息.txt" "${log}"
  if [[ "${exit_code}" -ne 0 ]]; then
    exit "${exit_code}"
  fi

  local model_name
  model_name="$(python - "${config}" <<'PY'
import sys
from lib.helpers.config_helper import load_config
print(load_config(sys.argv[1])["model_name"])
PY
)"
  local trained="${stage_dir}/${model_name}/checkpoint_best.pth"
  if [[ ! -f "${trained}" ]]; then
    echo "错误：${label}没有生成checkpoint_best.pth" >&2
    exit 7
  fi
  if [[ "${audit_type}" == "quality" ]]; then
    python tools/审计V23质量训练参数.py \
      --source "${CKPT_B}" --trained "${trained}" \
      2>&1 | tee "${stage_dir}/训练后参数审计.txt"
  else
    python tools/审计V23训练参数.py \
      --config "${config}" --source "${SOURCE_CHECKPOINT}" --trained "${trained}" \
      2>&1 | tee "${stage_dir}/训练后参数审计.txt"
  fi
}

run_g8() {
  require_empty_dir "${G8_DIR}"
  mkdir -p "${G8_DIR}"
  set -o pipefail
  python tools/诊断GPSD信息保真_G8.py \
    --config versions/V09_V08O加点式三维质量排序/config.yaml \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --groups 4 --scales 4 --num_samples 100 \
    --output_dir "${G8_DIR}" \
    2>&1 | tee "${G8_DIR}/运行.log"
  if ! grep -q '自动结论：\*\*通过，允许V23短程配对\*\*' "${G8_REPORT}"; then
    echo "G8未通过。停止V23正式配对。" >&2
    exit 6
  fi
}

require_g8_pass() {
  if [[ ! -f "${G8_REPORT}" ]] || ! grep -q '自动结论：\*\*通过，允许V23短程配对\*\*' "${G8_REPORT}"; then
    echo "错误：G8尚未通过。请先运行g8。" >&2
    exit 6
  fi
}

summarize_pair() {
  python tools/汇总V23分组视差预聚合.py \
    --control_log "${LOG_O}" --rdsa_log "${LOG_A}" --gpsd_log "${LOG_B}" \
    --output "${REPORT_PAIR}"
}

summarize_seeds() {
  python tools/汇总V23双随机种子.py \
    --o444_log "${LOG_O}" --a444_log "${LOG_A}" --b444_log "${LOG_B}" \
    --o445_log "${LOG_O_445}" --a445_log "${LOG_A_445}" --b445_log "${LOG_B_445}" \
    --output "${REPORT_SEEDS}"
}

run_latency() {
  if [[ ! -f "${REPORT_SEEDS}" ]] || ! grep -q 'GPSD配对增益复现：\*\*通过\*\*' "${REPORT_SEEDS}"; then
    echo "错误：GPSD双种子尚未通过，不启动正式时延。" >&2
    exit 6
  fi
  require_empty_dir "${LATENCY_DIR}"
  mkdir -p "${LATENCY_DIR}"
  run_one_latency() {
    local label="$1" config="$2" checkpoint="$3"
    python tools/测量QECR端到端时延.py \
      --config "${config}" --checkpoint "${checkpoint}" --warmup 20 --steps 100 \
      2>&1 | tee "${LATENCY_DIR}/${label}.log"
  }
  run_one_latency "第1组_V23O" "${CONFIG_O}" "${CKPT_O}"
  run_one_latency "第1组_V23B" "${CONFIG_B}" "${CKPT_B}"
  run_one_latency "第2组_V23B" "${CONFIG_B}" "${CKPT_B}"
  run_one_latency "第2组_V23O" "${CONFIG_O}" "${CKPT_O}"
  run_one_latency "第3组_V23O" "${CONFIG_O}" "${CKPT_O}"
  run_one_latency "第3组_V23B" "${CONFIG_B}" "${CKPT_B}"
  grep -HE 'GPU:|parameters:|median_ms:|mean_ms:|p90_ms:|fps_from_median:' \
    "${LATENCY_DIR}"/*.log | tee "${LATENCY_DIR}/汇总.txt"
  python tools/汇总V23时延.py \
    --latency_dir "${LATENCY_DIR}" --output "${LATENCY_DIR}/时延判定.md"
}

show_status() {
  echo "===== V23固定起点 ====="
  if [[ -f "${SOURCE_CHECKPOINT}" ]]; then
    ls -lh "${SOURCE_CHECKPOINT}"
    sha256sum "${SOURCE_CHECKPOINT}"
  else
    echo "尚未导入V09 checkpoint"
  fi
  echo "===== 进程 ====="
  pgrep -af '训练与评估_无蒸馏QECR.py|诊断GPSD信息保真_G8.py' || echo "当前没有V23训练或诊断进程"
  echo "===== 报告 ====="
  for report in "${G8_REPORT}" "${REPORT_PAIR}" "${REPORT_SEEDS}" "${LATENCY_DIR}/时延判定.md" "${REPORT_QUALITY}"; do
    [[ -f "${report}" ]] && echo "完成  ${report}" || echo "缺少  ${report}"
  done
  echo "===== 最近日志 ====="
  local latest
  latest="$(find "${RUN_ROOT}" -type f -name '控制台.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
  [[ -n "${latest}" ]] && tail -n 25 "${latest}" || true
}

if [[ "${MODE}" == "status" ]]; then
  show_status
  exit 0
fi

prepare_source_checkpoint

if [[ "${MODE}" == "preflight" || "${MODE}" == "all" ]]; then
  preflight
fi
if [[ "${MODE}" == "g8" || "${MODE}" == "all" ]]; then
  run_g8
fi
if [[ "${MODE}" == "smoke" || "${MODE}" == "all" ]]; then
  if [[ "${MODE}" != "all" ]]; then require_g8_pass; fi
  SMOKE_IDS="/root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train_v23_smoke.txt"
  [[ -f "${SMOKE_IDS}" ]] || head -n 24 /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train.txt > "${SMOKE_IDS}"
  run_stage "V23B_24样本冒烟" "${CONFIG_SMOKE}" "${DIR_SMOKE}" "${LOG_SMOKE}"
fi
if [[ "${MODE}" == "pair" || "${MODE}" == "control" || "${MODE}" == "rdsa" || "${MODE}" == "gpsd" || "${MODE}" == "all" ]]; then
  require_g8_pass
fi
if [[ "${MODE}" == "pair" || "${MODE}" == "control" || "${MODE}" == "all" ]]; then
  run_stage "V23O_原始相关公平控制" "${CONFIG_O}" "${DIR_O}" "${LOG_O}"
fi
if [[ "${MODE}" == "pair" || "${MODE}" == "rdsa" || "${MODE}" == "all" ]]; then
  run_stage "V23A_残差式视差空间微聚合" "${CONFIG_A}" "${DIR_A}" "${LOG_A}"
fi
if [[ "${MODE}" == "pair" || "${MODE}" == "gpsd" || "${MODE}" == "all" ]]; then
  run_stage "V23B_分组保真空间视差预聚合" "${CONFIG_B}" "${DIR_B}" "${LOG_B}"
fi
if [[ "${MODE}" == "pair" || "${MODE}" == "summary" || "${MODE}" == "all" ]]; then
  summarize_pair
fi
if [[ "${MODE}" == "seed445_pair" ]]; then
  [[ -f "${REPORT_PAIR}" ]] || summarize_pair
  if grep -q 'V23B自动判定：\*\*否决\*\*' "${REPORT_PAIR}"; then
    echo "V23B已否决，不运行第二随机种子。" >&2
    exit 6
  fi
  run_stage "V23O_seed445" "${CONFIG_O_445}" "${DIR_O_445}" "${LOG_O_445}"
  run_stage "V23A_seed445" "${CONFIG_A_445}" "${DIR_A_445}" "${LOG_A_445}"
  run_stage "V23B_seed445" "${CONFIG_B_445}" "${DIR_B_445}" "${LOG_B_445}"
  summarize_seeds
fi
if [[ "${MODE}" == "seed445_summary" ]]; then
  summarize_seeds
fi
if [[ "${MODE}" == "latency" ]]; then
  run_latency
fi
if [[ "${MODE}" == "quality" ]]; then
  if [[ ! -f "${LATENCY_DIR}/时延判定.md" ]] || ! grep -q '自动结论：\*\*通过\*\*' "${LATENCY_DIR}/时延判定.md"; then
    echo "错误：双种子与时延尚未全部通过，不启动质量重标定。" >&2
    exit 6
  fi
  run_stage "V23Q_GPSD查询级质量重标定" "${CONFIG_Q}" "${DIR_Q}" "${LOG_Q}" quality
  python tools/汇总V23质量重标定.py \
    --gpsd_log "${LOG_B}" --quality_log "${LOG_Q}" --output "${REPORT_QUALITY}"
fi
if [[ "${MODE}" == "quality_summary" ]]; then
  python tools/汇总V23质量重标定.py \
    --gpsd_log "${LOG_B}" --quality_log "${LOG_Q}" --output "${REPORT_QUALITY}"
fi
