#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-versions/V03_基线安全查询深度/config.yaml}"

cd "$PROJECT_ROOT"
echo "Training config: $CONFIG_PATH"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python tools/训练与评估_无蒸馏QECR.py --config "$CONFIG_PATH"
