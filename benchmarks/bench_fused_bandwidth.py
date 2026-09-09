"""Fused Add+RMSNorm 带宽对拍 — 5090 实测

对比: eager(add+rmsnorm) / torch.compile / 自研 CUDA 融合 kernel
有效字节: 读 x + 读 residual + 写 out (各 bf16) ≈ 6 B/elem + weight
配置: hidden=4096（与简历口径一致）
"""
import torch
import fused_add_rmsnorm

torch.manual_seed(0)
H = 4096
WEIGHT = torch.randn(H, dtype=torch.bfloat16, device="cuda")
rms_eps = 1e-6


def eager(x, residual):
    x = x + residual
    dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = xf * torch.rsqrt(var + rms_eps)
    return (xn.to(dtype) * WEIGHT), x


def run(fn, x, res, iters=200):
    for _ in range(20):
        fn(x, res)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn(x, res)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    gb = (6 * x.numel() + 2 * H) / 1e9
    return ms, gb / (ms / 1e3)


def main():
    print(f"{'tokens':>8} {'eager ms':>9} {'eager GB/s':>10} "
          f"{'compile ms':>10} {'compile GB/s':>12} "
          f"{'cuda ms':>8} {'cuda GB/s':>9} {'cuda/eager':>10}")
    for tokens in (1024, 4096, 8192, 16384, 32768):
        x = torch.randn(tokens, H, dtype=torch.bfloat16, device="cuda")
        res = torch.randn_like(x)

        ms_e, bw_e = run(eager, x, res)
        compiled = torch.compile(eager, mode="max-autotune-no-cudagraphs")
        ms_c, bw_c = run(compiled, x, res)

        out = torch.empty_like(x)
        res2 = torch.empty_like(x)

        def cuda_fn(x, res):
            return fused_add_rmsnorm.forward(x, res, WEIGHT, rms_eps), res2

        ms_k, bw_k = run(cuda_fn, x, res)
        print(f"{tokens:>8} {ms_e:>9.3f} {bw_e:>10.0f} "
              f"{ms_c:>10.3f} {bw_c:>12.0f} "
              f"{ms_k:>8.3f} {bw_k:>9.0f} {bw_k / bw_e:>9.2f}x")

    # 正确性抽查
    x = torch.randn(512, H, dtype=torch.bfloat16, device="cuda")
    res = torch.randn_like(x)
    ref, res_ref = eager(x, res)
    got = fused_add_rmsnorm.forward(x, res, WEIGHT, rms_eps)
    got = got[0] if isinstance(got, tuple) else got
    diff = (ref.float() - got.float()).abs().max().item()
    print(f"\n正确性: max|diff| = {diff:.2e} (bf16 容差内即对齐)")


if __name__ == "__main__":
    main()
