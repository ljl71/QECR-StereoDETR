#!/usr/bin/env bash
set -euo pipefail

# 使用已冻结的 V09 最佳权重，对 KITTI object test 的 7518 对双目图像
# 只运行一次最终推理，并生成 txt 位于 ZIP 根目录的官方提交包。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="versions/V09_KITTI官网测试提交/config.yaml"
TEST_ROOT="/root/autodl-tmp/datasets/KITTI/object/testing"
TEST_LIST="${TEST_ROOT}/ImageSets/test.txt"
SOURCE_CHECKPOINT="outputs/最终组合_V09/V09_V08O加点式三维质量排序/qecr_v09_v08o_pointwise_3d_quality/checkpoint_best.pth"
EXPECTED_SHA256="6b673fef2a54e2a5fbc44be731b24914770fffebaf1c20a0cacc43228dd316b4"
RUN_ROOT="outputs/KITTI官网测试_V09"
MODEL_DIR="${RUN_ROOT}/qecr_v09_kitti_test_submission"
RESULT_DIR="${MODEL_DIR}/outputs/data"
SUBMISSION_ZIP="${RUN_ROOT}/V09_KITTI_test_submission.zip"
LOG_FILE="${RUN_ROOT}/控制台.log"
INFO_FILE="${RUN_ROOT}/运行信息.txt"

if [[ "${CONDA_DEFAULT_ENV:-}" != "stereodetr-open" ]]; then
  echo "错误：请先激活 stereodetr-open 环境。" >&2
  exit 2
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NUMBA_CUDA_USE_NVIDIA_BINDING="${NUMBA_CUDA_USE_NVIDIA_BINDING:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

for required in "${CONFIG}" "${SOURCE_CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "错误：缺少文件 ${required}" >&2
    exit 3
  fi
done

actual_sha256="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "错误：V09 checkpoint SHA256 不符合已登记的最终模型。" >&2
  echo "期望：${EXPECTED_SHA256}" >&2
  echo "实际：${actual_sha256}" >&2
  exit 4
fi

mkdir -p "${TEST_ROOT}/ImageSets"
if [[ ! -f "${TEST_LIST}" ]]; then
  seq -f '%06g' 0 7517 > "${TEST_LIST}"
fi

python - "${TEST_ROOT}" "${TEST_LIST}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
test_list = Path(sys.argv[2])
ids = [line.strip() for line in test_list.read_text().splitlines() if line.strip()]
expected = [f"{index:06d}" for index in range(7518)]
if ids != expected:
    raise SystemExit("test.txt 必须按顺序完整包含 000000 到 007517")

missing = []
for sample_id in ids:
    for folder, suffix in (("image_2", ".png"), ("image_3", ".png"), ("calib", ".txt")):
        path = root / folder / f"{sample_id}{suffix}"
        if not path.is_file():
            missing.append(str(path))
            if len(missing) >= 20:
                break
    if len(missing) >= 20:
        break
if missing:
    raise SystemExit("KITTI test 数据不完整，示例缺失文件：\n" + "\n".join(missing))
print("KITTI test 数据检查通过：7518 对左右图像和标定文件完整")
PY

if [[ -e "${LOG_FILE}" || -e "${SUBMISSION_ZIP}" || -d "${RESULT_DIR}" ]]; then
  echo "错误：检测到已有 KITTI test 运行产物。" >&2
  echo "为避免混合或利用测试集反复调参，本脚本拒绝覆盖。" >&2
  exit 5
fi

mkdir -p "${MODEL_DIR}" "${RUN_ROOT}"
cp "${SOURCE_CHECKPOINT}" "${MODEL_DIR}/checkpoint_best.pth"
cp "${CONFIG}" "${RUN_ROOT}/配置快照.yaml"

start_ts="$(date +%s)"
{
  echo "purpose=final_kitti_test_submission"
  echo "training=none"
  echo "config=${CONFIG}"
  echo "checkpoint=${SOURCE_CHECKPOINT}"
  echo "checkpoint_sha256=${actual_sha256}"
  echo "start=$(date '+%F %T')"
  echo "python=$(command -v python)"
  python --version
  python - <<'PY'
import torch
print("pytorch={}".format(torch.__version__))
print("cuda_runtime={}".format(torch.version.cuda))
print("gpu={}".format(torch.cuda.get_device_name(0)))
PY
} > "${INFO_FILE}"

set +e
python tools/训练与评估_无蒸馏QECR.py \
  --evaluate_only \
  --config "${CONFIG}" \
  2>&1 | tee "${LOG_FILE}"
run_exit="${PIPESTATUS[0]}"
set -e

elapsed="$(( $(date +%s) - start_ts ))"
{
  echo "end=$(date '+%F %T')"
  echo "exit_code=${run_exit}"
  echo "elapsed_seconds=${elapsed}"
} | tee -a "${INFO_FILE}" "${LOG_FILE}"

if [[ "${run_exit}" -ne 0 ]]; then
  echo "KITTI test 推理失败，退出码=${run_exit}" >&2
  exit "${run_exit}"
fi

python - "${RESULT_DIR}" <<'PY'
from pathlib import Path
import math
import sys

result_dir = Path(sys.argv[1])
files = sorted(result_dir.glob("*.txt"))
expected_names = [f"{index:06d}.txt" for index in range(7518)]
if [path.name for path in files] != expected_names:
    raise SystemExit("结果文件必须完整包含 000000.txt 到 007517.txt")

classes = {"Car", "Pedestrian", "Cyclist"}
line_count = 0
for path in files:
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 16:
            raise SystemExit(f"{path}:{line_number} 不是 KITTI 16字段检测格式")
        if fields[0] not in classes:
            raise SystemExit(f"{path}:{line_number} 非法类别 {fields[0]}")
        values = [float(value) for value in fields[1:]]
        if not all(math.isfinite(value) for value in values):
            raise SystemExit(f"{path}:{line_number} 包含 NaN/Inf")
        line_count += 1
print(f"提交结果检查通过：7518个文件，{line_count}条检测，全部为KITTI 16字段格式")
PY

# 官方 object devkit 要求 000000.txt ... 007517.txt 直接位于 ZIP 根目录。
(
  cd "${RESULT_DIR}"
  zip -q "${PROJECT_ROOT}/${SUBMISSION_ZIP}" ./*.txt
)

zip_count="$(unzip -Z1 "${SUBMISSION_ZIP}" | wc -l)"
if [[ "${zip_count}" -ne 7518 ]]; then
  echo "错误：ZIP 内文件数为 ${zip_count}，期望 7518。" >&2
  exit 6
fi
if unzip -Z1 "${SUBMISSION_ZIP}" | grep -q '/'; then
  echo "错误：ZIP 中不应包含 data/ 等目录层级。" >&2
  exit 7
fi

sha256sum "${SUBMISSION_ZIP}" | tee "${SUBMISSION_ZIP}.sha256"
echo "KITTI最终提交包已生成：${PROJECT_ROOT}/${SUBMISSION_ZIP}"
echo "请勿根据官网测试结果继续调参或再次提交同一方法。"
