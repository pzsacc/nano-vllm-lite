"""
setup.py - CUDA 扩展编译配置

项目元数据已迁移至 pyproject.toml。
此文件仅保留 CUDA kernel 扩展的编译逻辑。

安装模式：
- `pip install -e .`: 安装 Python 包 + 编译 CUDA kernel 扩展
- 如果环境没有 CUDA toolkit，CUDA kernel 编译会被跳过，
  引擎自动 fallback 到 @torch.compile 实现
"""

import os
from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def get_cuda_extensions():
    if not CUDA_AVAILABLE:
        return []

    kernel_dir = os.path.join(os.path.dirname(__file__), "nano_vllm", "kernels")
    extra_compile_args = {
        "cxx": ["-O3"],
        "nvcc": ["-O3", "--use_fast_math"],
    }

    return [
        CUDAExtension(
            name="fused_add_rmsnorm",
            sources=[os.path.join(kernel_dir, "add_rmsnorm.cu")],
            extra_compile_args=extra_compile_args,
        ),
        CUDAExtension(
            name="fused_rope_cuda",
            sources=[os.path.join(kernel_dir, "inplace_rotary_embed.cu")],
            extra_compile_args=extra_compile_args,
        ),
    ]


setup(
    ext_modules=get_cuda_extensions(),
    cmdclass={"build_ext": BuildExtension} if CUDA_AVAILABLE else {},
)
