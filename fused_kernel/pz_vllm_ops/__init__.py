import torch

# 1. 尝试导入编译好的全局二进制扩展
try:
    import fused_rope_cuda
    import fused_add_rmsnorm
except ImportError as e:
    raise ImportError("未在系统环境中检测到编译好的二进制算子，请在根目录执行 pip install . --no-build-isolation") from e


# 2. 暴露给外部模型的 RoPE 接口
def apply_rope_inplace(q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor, cos_sin_cache: torch.Tensor):
    if not q.is_contiguous(): q = q.contiguous()
    if not k.is_contiguous(): k = k.contiguous()
    if not pos_ids.is_contiguous(): pos_ids = pos_ids.contiguous()
    if not cos_sin_cache.is_contiguous(): cos_sin_cache = cos_sin_cache.contiguous()

    if pos_ids.dtype != torch.int32:
        pos_ids = pos_ids.to(torch.int32)

    fused_rope_cuda.apply_fused_rope_inplace(q, k, pos_ids, cos_sin_cache)
    return q, k


# 3. 顺便把你的 RMSNorm 接口也在这里包装好，一专多能！
def apply_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    if not x.is_contiguous(): x = x.contiguous()
    if not residual.is_contiguous(): residual = residual.contiguous()
    if not weight.is_contiguous(): weight = weight.contiguous()

    return fused_add_rmsnorm.forward(x, residual, weight, eps)