"""Portable launcher for the calibration test without KITTI evaluation deps."""

import runpy
import sys
from pathlib import Path
from types import ModuleType


eval_name = "lib.datasets.kitti.kitti_eval_python.eval"
eval_stub = ModuleType(eval_name)
eval_stub.get_official_eval_result = lambda *args, **kwargs: None
eval_stub.get_distance_eval_result = lambda *args, **kwargs: None
sys.modules[eval_name] = eval_stub

common_name = "lib.datasets.kitti.kitti_eval_python.kitti_common"
sys.modules[common_name] = ModuleType(common_name)

runpy.run_path(
    str(Path(__file__).resolve().with_name("测试P2P3双目标定_最终.py")),
    run_name="__main__",
)

