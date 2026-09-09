import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 强行注入，干掉 OSError（本机 cu13 工具链）
os.environ.setdefault(
    "CUDA_HOME", "/root/autodl-tmp/site-packages/nvidia/cu13"
)

setup(
    name='pz_vllm_ops',
    version='1.0.0',
    # 告诉 Python，pz_vllm_ops 是一个合法的 Python 核心包
    packages=['pz_vllm_ops'],
    ext_modules=[
        CUDAExtension(
            name='fused_add_rmsnorm',
            sources=['add_rmsnorm.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK']}
        ),
        CUDAExtension(
            name='fused_rope_cuda',
            sources=['inplace_rotary_embed.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
