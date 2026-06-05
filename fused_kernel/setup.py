import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 强行注入，干掉 OSError
os.environ["CUDA_HOME"] = "/home/zheng-pingze/cuda-13.2"

setup(
    name='pz_vllm_ops',
    version='1.0.0',
    # 告诉 Python，pz_vllm_ops 是一个合法的 Python 核心包
    packages=['pz_vllm_ops'],
    ext_modules=[
        CUDAExtension(
            name='fused_add_rmsnorm',
            sources=['add_rmsnorm.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']}
        ),
        CUDAExtension(
            name='fused_rope_cuda',
            sources=['inplace_rotary_embed.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)