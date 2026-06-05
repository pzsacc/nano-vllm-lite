import torch
from torch import nn
'''#from pz_vllm_ops import apply_add_rmsnorm
#import triton
#import triton.language as tl
#from torch.library import wrap_triton
#DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]'''

class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x
    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)

'''@torch.compile
def add_rms_forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight,
    eps,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x, residual'''

'''# method 1: warmup
@triton.jit
def _add_rms_norm_kernel(
        x_ptr, residual_ptr, weight_ptr,
        x_out_ptr, residual_out_ptr,
        x_step, x_out_step, residual_step, residual_out_step, weight_step,
        n_rows, n_cols, eps,
        BLOCK_SIZE: tl.constexpr,
        num_stages: tl.constexpr
):
    PID = tl.program_id(0)
    row_step = tl.num_programs(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    w = tl.load(weight_ptr + offsets * weight_step, mask=mask, other=0.0).to(tl.float32)
    for row_idx in tl.range(PID, n_rows, row_step, num_stages=num_stages):
        row_x_ptr = x_ptr + row_idx * x_step
        row_res_ptr = residual_ptr + row_idx * residual_step
        row_x_out_ptr = x_out_ptr + row_idx * x_out_step
        row_res_out_ptr = residual_out_ptr + row_idx * residual_out_step
        x = tl.load(row_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(row_res_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x_r = x + r
        tl.store(row_res_out_ptr + offsets, x_r.to(tl.bfloat16), mask=mask)
        var = tl.sum(x_r * x_r, axis=0) / n_cols
        rstd = tl.math.rsqrt(var + eps)
        out = x_r * rstd * w
        tl.store(row_x_out_ptr + offsets, out.to(tl.bfloat16), mask=mask)
def add_rms_norm(x, residual, weight, eps):
    assert x.ndim == 2      # 确保输入矩阵二维
    assert x.is_contiguous()  # 确保按行优先内存
    N, hidden_layer = x.shape
    BLOCK_SIZE = triton.next_power_of_2(hidden_layer)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2

    x_out = torch.empty_like(x)
    residual_out = torch.empty_like(residual)

    kernel = _add_rms_norm_kernel.warmup(
        x, residual, weight,
        x_out, residual_out,
        x.stride(0), x_out.stride(0), residual.stride(0), residual_out.stride(0), weight.stride(0),
        N, hidden_layer, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,)
    )
    kernel._init_handles()
    n_regs_per_program = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared
    reg_occupancy = NUM_REGS // (n_regs_per_program * WARP_SIZE * num_warps)
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program
    programs_per_sm = min(reg_occupancy, sram_occupancy)
    num_programs = min(NUM_SM * programs_per_sm, N)

    def grid_fn(meta): return (num_programs, 1, 1)

    wrap_triton(_add_rms_norm_kernel)[grid_fn](
        x, residual, weight,
        x_out, residual_out,
        x.stride(0), x_out.stride(0), residual.stride(0), residual_out.stride(0), weight.stride(0),
        N, hidden_layer, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
    )
    return x_out, residual_out'''
    
'''
# method 2: no warmup    
@triton.jit
def _add_rms_norm_kernel(
    x_ptr, residual_ptr, weight_ptr,
    x_out_ptr, residual_out_ptr,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    x_r = x + r
    tl.store(residual_out_ptr + row * N + offsets, x_r.to(tl.bfloat16), mask=mask)
    var = tl.sum(x_r * x_r, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    out = x_r * rstd * w
    tl.store(x_out_ptr + row * N + offsets, out.to(tl.bfloat16), mask=mask)
def add_rms_norm(x, residual, weight, eps):
    M, N = x.shape
    BLOCK_N = triton.next_power_of_2(N)  # 4096 → 4096, 8192 → 8192
    x_out = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    _add_rms_norm_kernel[(M,)](x, residual, weight, x_out, residual_out, N, eps, BLOCK_N)
    return x_out, residual_out
'''

'''@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['hidden_layer'],
        x_vals=[128*i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['cuda', 'torch'],
        line_names=['CUDA', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='RMSNorm_Add_performance_CUDA',
        args={'N':128}
    )
)
def benchmark(N, hidden_layer, provider):
    x = torch.randn(N, hidden_layer, device=DEVICE, dtype=torch.bfloat16)
    residual = torch.randn(N, hidden_layer, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(hidden_layer, device=DEVICE, dtype=torch.bfloat16)
    eps = 1e-6
    x_input = x.clone()
    res_input = residual.clone()

    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'cuda':
        ms = triton.testing.do_bench(lambda: apply_add_rmsnorm(x_input, res_input, weight, eps))

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: add_rms_forward(x_input, res_input, weight, eps))

    element_bytes = x.element_size()
    matrix_elems = x.numel()
    total_bytes = 4 * matrix_elems * element_bytes
    total_bytes += weight.numel() * weight.element_size()
    gbps = total_bytes * 1e-9 / (ms * 1e-3)
    return gbps

def test_for_kernel(size: tuple, atol=1e-2, rtol=1e-2):
    assert type(size) is tuple and len(size) == 2
    torch.manual_seed(0)

    x = torch.randn(size[0], size[1], device=DEVICE, dtype=torch.bfloat16)
    residual = torch.randn(size[0], size[1], device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(size[1], device=DEVICE, dtype=torch.bfloat16)
    eps = 1e-6

    z_tri = apply_add_rmsnorm(x.clone(), residual.clone(), weight, eps)
    z_ref = add_rms_forward(x.clone(), residual.clone(), weight, eps)
    torch.testing.assert_close(z_tri[0], z_ref[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(z_tri[1], z_ref[1], atol=atol, rtol=rtol)
    print("passed")
if __name__ == '__main__':
    test_for_kernel((128, 4096))
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark.run(save_path='.', print_data=False)'''

