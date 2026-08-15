"""Local dependency-free launcher for the P2/P3 calibration unit test."""

import runpy
import sys
from pathlib import Path
from types import ModuleType


package_name = "lib.datasets.kitti.kitti_eval_python"
package = ModuleType(package_name)
package.__path__ = []
sys.modules[package_name] = package

eval_name = package_name + ".eval"
eval_module = ModuleType(eval_name)
eval_module.get_official_eval_result = lambda *args, **kwargs: None
eval_module.get_distance_eval_result = lambda *args, **kwargs: None
sys.modules[eval_name] = eval_module
package.eval = eval_module

common_name = package_name + ".kitti_common"
common_module = ModuleType(common_name)
sys.modules[common_name] = common_module
package.kitti_common = common_module

runpy.run_path(
    str(Path(__file__).resolve().with_name("测试P2P3双目标定_最终.py")),
    run_name="__main__",
)

