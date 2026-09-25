#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/运行MQD论文补充实验.sh preflight
#   bash scripts/运行MQD论文补充实验.sh smoke
#   bash scripts/运行MQD论文补充实验.sh shared
#   bash scripts/运行MQD论文补充实验.sh boundary
#   bash scripts/运行MQD论文补充实验.sh seed445
#   bash scripts/运行MQD论文补充实验.sh analysis
#   bash scripts/运行MQD论文补充实验.sh latency
#   bash scripts/运行MQD论文补充实验.sh summary
#   bash scripts/运行MQD论文补充实验.sh all
#   bash scripts/运行MQD论文补充实验.sh status

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|shared|boundary|seed445|analysis|latency|summary|all|status) ;;
  *) echo "错误：未知模式 ${MODE}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

V08O_CKPT="outputs/第二创新点_V08空间投影对齐/V08O_公平控制组/qecr_v08o_geometry_control/checkpoint_best.pth"
V09_CKPT="outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
V08O_LOG="outputs/第二创新点_V08空间投影对齐/V08O_公平控制组/控制台.log"
V09_LOG="outputs/最终组合_V09/V09_V08O加点式三维质量排序/控制台.log"
CONFIG_ROOT="configs/MQD论文补充实验"
OUTPUT_ROOT="outputs/MQD论文补充实验"
SMOKE_LIST="/root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train_mqd_paper_smoke.txt"

declare -A CONFIGS=(
  [S1]="${CONFIG_ROOT}/S1_Hungarian匹配查询监督.yaml"
  [S2]="${CONFIG_ROOT}/S2_全查询同类监督无尾部平衡.yaml"
  [S3]="${CONFIG_ROOT}/S3_全查询同类监督加尾部平衡.yaml"
  [B1]="${CONFIG_ROOT}/B1_普通Car残差.yaml"
  [B2]="${CONFIG_ROOT}/B2_零初始化有界残差.yaml"
  [B3]="${CONFIG_ROOT}/B3_边界加权决策场.yaml"
  [B4]="${CONFIG_ROOT}/B4_完整边界决策场.yaml"
  [B4S]="${CONFIG_ROOT}/B4_完整边界决策场_种子445.yaml"
)
declare -A MODEL_NAMES=(
  [S1]="mqd_s1_matched_query_supervision"
  [S2]="mqd_s2_all_query_without_tail_balance"
  [S3]="mqd_s3_all_query_tail_balanced"
  [B1]="mqd_b1_plain_car_residual"
  [B2]="mqd_b2_bounded_zero_init_residual"
  [B3]="mqd_b3_boundary_weighted_field"
  [B4]="mqd_b4_complete_boundary_decision_field"
  [B4S]="mqd_b4_complete_boundary_decision_field_seed445"
)
declare -A DIRS=(
  [S1]="${OUTPUT_ROOT}/S1_Hungarian匹配查询监督"
  [S2]="${OUTPUT_ROOT}/S2_全查询同类监督无尾部平衡"
  [S3]="${OUTPUT_ROOT}/S3_全查询同类监督加尾部平衡"
  [B1]="${OUTPUT_ROOT}/B1_普通Car残差"
  [B2]="${OUTPUT_ROOT}/B2_零初始化有界残差"
  [B3]="${OUTPUT_ROOT}/B3_边界加权决策场"
  [B4]="${OUTPUT_ROOT}/B4_完整边界决策场"
  [B4S]="${OUTPUT_ROOT}/B4_完整边界决策场_种子445"
)

die() { echo "错误：$*" >&2; exit 3; }

require_environment() {
  [[ "${CONDA_DEFAULT_ENV:-}" == "stereodetr-open" ]] \
    || die "请先执行 source /root/miniconda3/bin/activate stereodetr-open"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export NUMBA_CUDA_USE_NVIDIA_BINDING="${NUMBA_CUDA_USE_NVIDIA_BINDING:-1}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=8
  fi
  export PYTHONUNBUFFERED=1
}

require_file() { [[ -f "$1" ]] || die "缺少文件：$1"; }

checkpoint_for() {
  local key="$1"
  echo "${DIRS[$key]}/${MODEL_NAMES[$key]}/checkpoint_best.pth"
}

run_preflight() {
  python tools/检查MQD论文补充实验.py
  python tools/测试3D质量排序.py
  python -m py_compile \
    lib/models/monodetr/quality_ranking.py \
    lib/models/monodetr/stereodetr.py \
    tools/分析MQD度量一致性.py \
    tools/汇总MQD论文补充实验.py \
    tools/汇总MQD完整时延.py \
    tools/审计MQD补充实验权重.py
  require_file "${V08O_CKPT}"
  require_file "${V09_CKPT}"
  require_file /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train.txt
  require_file /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/val.txt
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA不可用")
p = torch.cuda.get_device_properties(0)
print("GPU: {}，显存{:.2f} GiB".format(p.name, p.total_memory / 1024 ** 3))
PY
  echo "MQD论文补充实验预检通过"
}

run_smoke() {
  local smoke_checkpoint
  smoke_checkpoint="${OUTPUT_ROOT}/冒烟测试/mqd_paper_evidence_smoke/checkpoint_final.pth"
  mkdir -p "$(dirname "${SMOKE_LIST}")" "${OUTPUT_ROOT}/冒烟测试"
  head -n 32 /root/autodl-tmp/datasets/KITTI/object/training/ImageSets/train.txt > "${SMOKE_LIST}"
  python tools/训练与评估_无蒸馏QECR.py \
    --config "${CONFIG_ROOT}/冒烟测试.yaml" --batch_size 8 --gpu_ids 0 --amp \
    2>&1 | tee "${OUTPUT_ROOT}/冒烟测试/控制台.log"
  require_file "${smoke_checkpoint}"
  python tools/审计MQD补充实验权重.py \
    --source "${V09_CKPT}" --trained "${smoke_checkpoint}" --scope car \
    --output "${OUTPUT_ROOT}/冒烟测试/训练后参数审计.txt"
}

train_one() {
  local key="$1" scope="$2" source="$3"
  local checkpoint
  checkpoint="$(checkpoint_for "${key}")"
  mkdir -p "${DIRS[$key]}"
  if [[ ! -f "${checkpoint}" ]]; then
    python tools/训练与评估_无蒸馏QECR.py \
      --config "${CONFIGS[$key]}" --batch_size 12 --gpu_ids 0 --amp \
      2>&1 | tee "${DIRS[$key]}/控制台.log"
  else
    echo "${key}已有checkpoint_best，跳过训练"
  fi
  require_file "${checkpoint}"
  python tools/审计MQD补充实验权重.py \
    --source "${source}" --trained "${checkpoint}" --scope "${scope}" \
    --output "${DIRS[$key]}/训练后参数审计.txt"
}

run_shared() {
  for key in S1 S2 S3; do train_one "${key}" shared "${V08O_CKPT}"; done
}

run_boundary() {
  for key in B1 B2 B3 B4; do train_one "${key}" car "${V09_CKPT}"; done
}

run_seed445() { train_one B4S car "${V09_CKPT}"; }

run_analysis() {
  local b4_ckpt
  b4_ckpt="$(checkpoint_for B4)"
  require_file "${b4_ckpt}"
  python tools/分析MQD度量一致性.py \
    --variant "StereoDETR::versions/V08O_几何对齐短程对照/config.yaml::${V08O_CKPT}" \
    --variant "Shared-MQD::versions/V09_V08O加点式三维质量排序/config.yaml::${V09_CKPT}" \
    --variant "Complete-MQD::${CONFIGS[B4]}::${b4_ckpt}" \
    --output_dir "${OUTPUT_ROOT}/度量一致性分析" --topk 50
}

measure_one() {
  local round="$1" variant="$2" config="$3" checkpoint="$4"
  python tools/测量QECR端到端时延.py \
    --config "${config}" --checkpoint "${checkpoint}" --warmup 50 --steps 200 \
    > "${OUTPUT_ROOT}/完整时延/第${round}组_${variant}.log" 2>&1
}

run_latency() {
  local b4_ckpt round
  b4_ckpt="$(checkpoint_for B4)"
  require_file "${b4_ckpt}"
  mkdir -p "${OUTPUT_ROOT}/完整时延"
  for round in 1 2 3 4 5 6; do
    if (( round % 2 == 1 )); then
      measure_one "${round}" baseline versions/V08O_几何对齐短程对照/config.yaml "${V08O_CKPT}"
      measure_one "${round}" shared versions/V09_V08O加点式三维质量排序/config.yaml "${V09_CKPT}"
      measure_one "${round}" complete "${CONFIGS[B4]}" "${b4_ckpt}"
    else
      measure_one "${round}" complete "${CONFIGS[B4]}" "${b4_ckpt}"
      measure_one "${round}" shared versions/V09_V08O加点式三维质量排序/config.yaml "${V09_CKPT}"
      measure_one "${round}" baseline versions/V08O_几何对齐短程对照/config.yaml "${V08O_CKPT}"
    fi
  done
  python tools/汇总MQD完整时延.py \
    --root "${OUTPUT_ROOT}/完整时延" \
    --output "${OUTPUT_ROOT}/完整时延/完整时延报告.md"
}

run_summary() {
  local arguments=()
  [[ -f "${V08O_LOG}" ]] && arguments+=(--entry "S0-StereoDETR=${V08O_LOG}")
  [[ -f "${V09_LOG}" ]] && arguments+=(--entry "B0-Shared-MQD=${V09_LOG}")
  local key
  for key in S1 S2 S3 B1 B2 B3 B4 B4S; do
    [[ -f "${DIRS[$key]}/控制台.log" ]] \
      && arguments+=(--entry "${key}=${DIRS[$key]}/控制台.log")
  done
  (( ${#arguments[@]} > 0 )) || die "没有可汇总日志"
  python tools/汇总MQD论文补充实验.py "${arguments[@]}" \
    --output "${OUTPUT_ROOT}/MQD论文补充实验汇总.md"
}

show_status() {
  echo "===== MQD论文补充实验状态 ====="
  local key checkpoint
  for key in S1 S2 S3 B1 B2 B3 B4 B4S; do
    checkpoint="$(checkpoint_for "${key}")"
    [[ -f "${checkpoint}" ]] && echo "完成  ${key}" || echo "未完成  ${key}"
  done
  [[ -f "${OUTPUT_ROOT}/度量一致性分析/度量一致性报告.md" ]] \
    && echo "完成  度量一致性分析" || echo "未完成  度量一致性分析"
  [[ -f "${OUTPUT_ROOT}/完整时延/完整时延报告.md" ]] \
    && echo "完成  完整时延" || echo "未完成  完整时延"
}

require_environment
if [[ "${MODE}" == status ]]; then show_status; exit 0; fi
run_preflight
case "${MODE}" in
  preflight) ;;
  smoke) run_smoke ;;
  shared) run_shared ;;
  boundary) run_boundary ;;
  seed445) run_seed445 ;;
  analysis) run_analysis ;;
  latency) run_latency ;;
  summary) run_summary ;;
  all)
    run_smoke
    run_shared
    run_boundary
    run_seed445
    run_analysis
    run_latency
    run_summary
    ;;
esac
show_status
