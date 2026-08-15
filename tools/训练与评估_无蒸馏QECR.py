"""Final AutoDL entry point using the runtime-tested QECR module."""

from pathlib import Path
import runpy

from QECR运行时引导 import install_runtime_module


install_runtime_module()
runpy.run_path(
    str(Path(__file__).resolve().with_name("训练与评估_QECR.py")),
    run_name="__main__",
)

