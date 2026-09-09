"""
setup.py - 项目安装配置

安装模式：
- `pip install -e .`: 安装 Python 包 + 编译 CUDA kernel 扩展
- 如果环境没有 CUDA toolkit，CUDA kernel 编译会被跳过，
  引擎自动 fallback 到 @torch.compile 实现

依赖：
- torch >= 2.0
- triton >= 2.0
- flash-attn >= 2.5
- transformers
- xxhash
- safetensors
"""

import os
from setuptools import setup, find_packages

# 尝试导入 CUDA 扩展构建工具
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def get_cuda_extensions():
    """构建 CUDA 扩展列表（如果 CUDA 可用）"""
    if not CUDA_AVAILABLE:
        return []

    kernel_dir = os.path.join(os.path.dirname(__file__), "nano_vllm", "kernels")
    extra_compile_args = {
        "cxx": ["-O3"],
        "nvcc": ["-O3", "--use_fast_math"],
    }

    extensions = [
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
    return extensions


setup(
    name="nano-vllm",
    version="0.3.0",
    description="轻量级高性能 LLM 推理引擎，支持 PagedAttention、FP8 KV Cache、Chunked Prefill",
    author="pzsacc",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "triton>=2.0",
        "flash-attn>=2.5",
        "transformers",
        "safetensors",
        "xxhash",
        "numpy",
        "tqdm",
    ],
    extras_require={
        "server": ["fastapi", "uvicorn", "sse-starlette"],
        "dev": ["pytest", "triton"],
    },
    ext_modules=get_cuda_extensions(),
    cmdclass={"build_ext": BuildExtension} if CUDA_AVAILABLE else {},
)
