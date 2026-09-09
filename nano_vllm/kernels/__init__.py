"""
kernels/__init__.py - 自定义 CUDA Kernel 扩展包

提供手写 CUDA kernel 的 Python 封装：
- apply_add_rmsnorm: 融合 Residual Add + RMSNorm（替代 torch.compile 版本）
- apply_rope_inplace: In-place RoPE 旋转位置编码（零额外内存分配）

这些 kernel 需要通过 `pip install -e .` 编译安装后才可用。
未安装时引擎自动 fallback 到 @torch.compile 实现。
"""

_CUDA_AVAILABLE = False

try:
    import fused_add_rmsnorm
    import fused_rope_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    pass


def is_cuda_kernels_available() -> bool:
    """检查自定义 CUDA kernel 是否已编译安装"""
    return _CUDA_AVAILABLE


def apply_rope_inplace(q, k, pos_ids, cos_sin_cache):
    """In-place RoPE 旋转位置编码

    直接在 Q/K tensor 上原地修改，无需分配新 tensor。
    2D grid (token × head)，每个线程处理一对维度。

    Args:
        q: [num_tokens, num_q_heads, head_dim] query tensor（原地修改）
        k: [num_tokens, num_k_heads, head_dim] key tensor（原地修改）
        pos_ids: [num_tokens] 位置索引（int32）
        cos_sin_cache: [max_position, head_dim] 预计算的 cos/sin 表

    Returns:
        (q, k) 原地修改后的引用
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernel 未安装，请运行 `pip install -e .` 编译")
    q = q.contiguous()
    k = k.contiguous()
    pos_ids = pos_ids.contiguous().to(dtype=q.new_zeros(1, dtype=int).dtype)
    cos_sin_cache = cos_sin_cache.contiguous()
    fused_rope_cuda.apply_fused_rope_inplace(q, k, pos_ids, cos_sin_cache)
    return q, k


def apply_add_rmsnorm(x, residual, weight, eps=1e-5):
    """融合 Residual Add + RMSNorm（CUDA kernel 实现）

    单 kernel 完成 residual += x 和 RMSNorm(residual)，
    通过 shared memory tree reduction 高效计算行方差。

    Args:
        x: [N, hidden_size] 输入
        residual: [N, hidden_size] 残差（原地更新）
        weight: [hidden_size] RMSNorm 缩放权重
        eps: 数值稳定常数

    Returns:
        (out, residual): 归一化输出和更新后的残差
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernel 未安装，请运行 `pip install -e .` 编译")
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()
    return fused_add_rmsnorm.forward(x, residual, weight, eps)
