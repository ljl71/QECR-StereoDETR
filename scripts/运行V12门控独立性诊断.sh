#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "stereodetr-open" ]]; then
  echo "请先激活stereodetr-open环境" >&2
  exit 2
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NUMBA_CUDA_USE_NVIDIA_BINDING="${NUMBA_CUDA_USE_NVIDIA_BINDING:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

V09="outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
D444="outputs/第二创新点_V12分组相关门控/V12A_s4十六组轻量门控/qecr_v12a_s4_group16_light_gate/checkpoint_best.pth"
D445="outputs/第二创新点_V12分组相关门控/第二随机种子_445/V12A_s4十六组轻量门控/qecr_v12a_seed445_s4_group16_light_gate/checkpoint_best.pth"
OUT="outputs/第二创新点_V12分组相关门控/门控独立性移植诊断"
H0="${OUT}/H0_V09冻结复评"
H1="${OUT}/H1_种子444门控移植"
H2="${OUT}/H2_种子445门控移植"

for checkpoint in "${V09}" "${D444}" "${D445}"; do
  [[ -f "${checkpoint}" ]] || { echo "缺少checkpoint：${checkpoint}" >&2; exit 4; }
done
if [[ -d "${OUT}" ]] && [[ -n "$(find "${OUT}" -mindepth 1 -print -quit)" ]]; then
  echo "诊断目录已有内容，为避免覆盖未启动：${OUT}" >&2
  exit 3
fi

python tools/汇总V12门控移植诊断.py --self_test

python tools/构建V12门控移植权重.py \
  --base "${V09}" \
  --output "${H0}/qecr_v12h0_v09_frozen_evaluation/checkpoint_best.pth"
python tools/构建V12门控移植权重.py \
  --base "${V09}" --donor "${D444}" \
  --output "${H1}/qecr_v12h1_seed444_gate_transplant/checkpoint_best.pth"
python tools/构建V12门控移植权重.py \
  --base "${V09}" --donor "${D445}" \
  --output "${H2}/qecr_v12h2_seed445_gate_transplant/checkpoint_best.pth"

python tools/训练与评估_无蒸馏QECR.py --evaluate_only \
  --config configs/AutoDL_V12H0_V09冻结复评.yaml \
  2>&1 | tee "${H0}/评估.log"
python tools/训练与评估_无蒸馏QECR.py --evaluate_only \
  --config configs/AutoDL_V12H1_种子444门控移植.yaml \
  2>&1 | tee "${H1}/评估.log"
python tools/训练与评估_无蒸馏QECR.py --evaluate_only \
  --config configs/AutoDL_V12H2_种子445门控移植.yaml \
  2>&1 | tee "${H2}/评估.log"

python tools/汇总V12门控移植诊断.py \
  --control_log "${H0}/评估.log" \
  --seed444_log "${H1}/评估.log" \
  --seed445_log "${H2}/评估.log" \
  --output "${OUT}/门控独立性诊断报告.md"
