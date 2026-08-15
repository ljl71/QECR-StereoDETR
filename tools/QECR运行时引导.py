"""Install the runtime-tested epipolar module under the stable import name."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def install_runtime_module():
    stable_name = "lib.models.monodetr.query_epipolar"
    runtime_path = (
        ROOT
        / "lib"
        / "models"
        / "monodetr"
        / "query_epipolar_runtime.py"
    )
    spec = importlib.util.spec_from_file_location(stable_name, runtime_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[stable_name] = module
    return module

