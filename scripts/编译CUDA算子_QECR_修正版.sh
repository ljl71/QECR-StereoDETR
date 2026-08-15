#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPS_DIR="$PROJECT_ROOT/lib/models/monodetr/ops"

echo "PyTorch/CUDA information:"
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-auto-detect}"
cd "$OPS_DIR"
python setup_qecr.py build install

cd "$PROJECT_ROOT"
python - <<'PY'
import torch
import MultiScaleDeformableAttention

print("torch shared libraries loaded from:", torch.__file__)
print("MultiScaleDeformableAttention import passed")
PY
