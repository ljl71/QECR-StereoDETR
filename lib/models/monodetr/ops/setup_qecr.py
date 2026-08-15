"""Build MultiScaleDeformableAttention for the GPU visible at install time."""

import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDAExtension,
    CUDA_HOME,
    BuildExtension,
)


def get_extensions():
    if not torch.cuda.is_available() or CUDA_HOME is None:
        raise RuntimeError(
            "CUDA and nvcc are required to build MultiScaleDeformableAttention"
        )
    this_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(this_dir, "src")
    sources = (
        glob.glob(os.path.join(source_dir, "*.cpp"))
        + glob.glob(os.path.join(source_dir, "cpu", "*.cpp"))
        + glob.glob(os.path.join(source_dir, "cuda", "*.cu"))
    )
    return [
        CUDAExtension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=[source_dir],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args={
                "cxx": [],
                "nvcc": [
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ]


setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    packages=find_packages(exclude=("configs", "tests")),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)

