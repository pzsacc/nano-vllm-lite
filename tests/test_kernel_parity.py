"""
tests/test_kernel_parity.py - 手写 CUDA Kernel 数值一致性测试 (需 GPU)

对应 docs/05-cuda-kernels.md。
这个测试防的是: kernel 实现语义错误（如 residual 原地更新语义与参考实现不一致）。
kernel 未编译时全部 skip。

运行: pytest tests/test_kernel_parity.py  (需 GPU + 已编译扩展)
CI 中跳过: pytest -m "not gpu"
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("fused_add_rmsnorm")

from nano_vllm.kernels import apply_add_rmsnorm  # noqa: E402


def reference_add_rmsnorm(x, residual, weight, eps):
    """与 layers/layernorm.py 中 @torch.compile 版本一致的参考实现"""
    input_dtype = x.dtype
    residual = (x.float() + residual.float()).to(input_dtype)
    xr = residual.float()
    var = xr.pow(2).mean(-1, keepdim=True)
    out = (xr * torch.rsqrt(var + eps) * weight.float()).to(input_dtype)
    return out, residual


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [1024, 4096])
def test_add_rmsnorm_parity(dtype, hidden):
    """CUDA kernel vs 参考实现: residual 逐位一致, 输出 bf16 1-ulp 内"""
    torch.manual_seed(42)
    n, eps = 256, 1e-6
    x = (torch.randn(n, hidden, device="cuda") * 2).to(dtype)
    res0 = (torch.randn(n, hidden, device="cuda") * 2).to(dtype)
    w = torch.randn(hidden, device="cuda").to(dtype) * 0.5 + 1.0

    out_cuda, res_cuda = apply_add_rmsnorm(x.clone(), res0.clone(), w, eps)
    out_ref, res_ref = reference_add_rmsnorm(x, res0, w, eps)

    # residual 原地更新必须与参考完全一致（否则残差流跨层漂移）
    assert torch.equal(res_cuda, res_ref), "residual 原地更新不一致"
    # 输出允许不同舍入顺序带来的差异
    # (kernel 用 fast_math + rsqrtf, 参考实现逐 op 舍入; 4096 长行归约累积更大)
    d = (out_cuda.float() - out_ref.float()).abs().max().item()
    tol = 8e-2 if dtype == torch.bfloat16 else 8e-3
    assert d < tol, f"输出偏差过大: {d:.4e} (tol={tol})"


def test_batch_size_1_and_odd():
    """边界形状: batch=1 与非对齐 batch"""
    torch.manual_seed(0)
    for n in (1, 7):
        x = torch.randn(n, 512, device="cuda").to(torch.bfloat16)
        r = torch.randn(n, 512, device="cuda").to(torch.bfloat16)
        w = torch.ones(512, device="cuda").to(torch.bfloat16)
        out_cuda, res_cuda = apply_add_rmsnorm(x.clone(), r.clone(), w, 1e-6)
        out_ref, res_ref = reference_add_rmsnorm(x, r, w, 1e-6)
        assert torch.equal(res_cuda, res_ref)
