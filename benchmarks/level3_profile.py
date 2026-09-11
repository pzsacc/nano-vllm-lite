"""
benchmarks/level3_profile.py - L3 算子与 profile

回答"慢在哪一行"的问题。三个工具：

- profiler: torch.profiler 生成 Chrome Trace (prefill/decode 分开)
  用法: chrome://tracing 加载 JSON, 或 tensorboard --logdir traces/
- bandwidth: fused Add+RMSNorm 带宽对拍 (eager / compile / CUDA kernel)
  教学点: 算子级 GB/s 推导 —— 有效字节 = 读x + 读residual + 写out ≈ 6B/elem
- nsys: 打印 Nsight Systems / Nsight Compute 的命令模板与解读指南
  (nsys/ncu 需原生二进制支持, 这里提供可复制的命令与 nvtx 标记说明)
"""
import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def profile_engine(model_path: str, input_len: int, output_len: int,
                   num_prompts: int, trace_dir: str):
    """torch.profiler: 分别 profile prefill 与 decode, 导出 Chrome Trace"""
    from nano_vllm import LLM, SamplingParams
    from torch.profiler import (profile, ProfilerActivity,
                                tensorboard_trace_handler)

    os.makedirs(trace_dir, exist_ok=True)
    llm = LLM(model_path, max_model_len=max(input_len + output_len, 2048),
              enable_fp8_kvcache=False)
    prompts = [" hi " * (input_len // 4)] * num_prompts
    sp = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)

    # 预热 (触发 compile/graph capture)
    llm.generate(prompts[:1], SamplingParams(temperature=1.0, max_tokens=8, ignore_eos=True))

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    print(f"\n[profile] prefill: {num_prompts} x {input_len} tok")
    with profile(activities=activities) as prof:
        llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=1, ignore_eos=True))
    path = os.path.join(trace_dir, "prefill.json.gz")
    prof.export_chrome_trace(path)
    _print_top_kernels(prof, "PREFILL")

    print(f"\n[profile] decode: {output_len} steps")
    with profile(activities=activities) as prof:
        llm.generate(prompts, sp)
    path = os.path.join(trace_dir, "decode.json.gz")
    prof.export_chrome_trace(path)
    _print_top_kernels(prof, "DECODE")

    print(f"\nTrace 已导出: {trace_dir}/")
    print("查看: chrome://tracing 加载 .json.gz, 或 tensorboard --logdir traces/")
    print("教学点: 在 trace 中找 GPU gap —— decode 阶段 CPU launch 开销若形成空白,")
    print("        即 CUDA Graph 要解决的问题 (见 docs/01)。")


def _print_top_kernels(prof, label: str):
    print(f"\n[{label}] Top GPU kernels (按 CUDA 时间):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))


def bandwidth_benchmark():
    """Fused Add+RMSNorm 带宽对拍: eager / compile / CUDA kernel"""
    from torch import nn

    try:
        from nano_vllm.kernels import apply_add_rmsnorm, is_cuda_kernels_available
    except Exception:
        is_cuda_kernels_available = lambda: False  # noqa: E731
    if not is_cuda_kernels_available():
        print("CUDA kernel 未编译 (pip install -e .), 跳过 kernel 对拍")
        return

    torch.manual_seed(0)
    H = 4096
    weight = torch.randn(H, dtype=torch.bfloat16, device="cuda")
    eps = 1e-6

    def eager(x, residual):
        r = (x.float() + residual.float()).to(x.dtype)
        xr = r.float()
        var = xr.pow(2).mean(-1, keepdim=True)
        return (xr * torch.rsqrt(var + eps) * weight.float()).to(x.dtype), r

    @torch.compile
    def compiled(x, residual):
        return eager(x, residual)

    def bench(fn, x, r, iters=200):
        for _ in range(20):
            fn(x, r)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(x, r)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    print("=" * 72)
    print("L3 算子带宽对拍: Fused Add+RMSNorm (hidden=4096, bf16)")
    print("=" * 72)
    print(f"{'tokens':>8}{'eager ms':>10}{'compile ms':>11}{'cuda ms':>9}"
          f"{'eager GB/s':>11}{'cuda GB/s':>10}{'加速比':>7}")
    print("-" * 72)
    for n in (1024, 4096, 16384):
        x = torch.randn(n, H, dtype=torch.bfloat16, device="cuda")
        r = torch.randn(n, H, dtype=torch.bfloat16, device="cuda")
        t_e = bench(eager, x, r)
        t_c = bench(compiled, x, r)
        t_k = bench(lambda a, b: apply_add_rmsnorm(a, b, weight, eps), x, r)
        # 有效字节: 读 x + 读 residual + 写 out + 写 residual = 8B/elem + weight
        bytes_moved = n * H * 2 * 4 + H * 2
        gbps = lambda t: bytes_moved / (t / 1e3) / 1e9  # noqa: E731
        print(f"{n:>8}{t_e:>10.3f}{t_c:>11.3f}{t_k:>9.3f}"
              f"{gbps(t_e):>11.0f}{gbps(t_k):>10.0f}{t_e / t_k:>6.2f}x")
    print("\n教学点: RMSNorm 是 memory-bound 算子, 收益上限 = 减少的访存次数;")
    print("        compile 与手写 kernel 大致持平说明融合已是主要收益 (见 docs/05)。")


NSYS_GUIDE = """
Nsight Systems / Compute 使用指南 (需本机安装 nsight-systems / nsight-compute)

# 全链路 timeline (含 nvtx 标记: engine step 已有 push/pop)
nsys profile -o report --trace=cuda,nvtx,osrt \\
    python examples/offline_inference.py --model /path/to/Qwen3-0.6B

# 查看: nsys-ui report.nsys-rep  或导出统计
nsys stats --report cuda_gpu_kern_sum report.nsys-rep

# 单 kernel 深挖 (如 fused_add_rmsnorm)
ncu --kernel-name regex:add_rmsnorm --set full \\
    python examples/offline_inference.py --model /path/to/Qwen3-0.6B

解读要点:
1. nsys timeline: 找 decode 步骤间的 gap → CPU launch 开销 (CUDA Graph 的靶子)
2. ncu speed-of-light: SOL memory > 80% 说明访存已打满, 优化空间在融合而非单 kernel
3. ncu launch stats: 占用率低 → block/grid 配置问题
"""


def main():
    parser = argparse.ArgumentParser(description="L3 算子与 profile")
    parser.add_argument("--model", required=False, default=None)
    parser.add_argument("--tool", choices=["profiler", "bandwidth", "nsys", "all"],
                        default="profiler")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--trace-dir", default="traces")
    args = parser.parse_args()

    if args.tool == "bandwidth" or args.tool == "nsys":
        if args.tool == "bandwidth":
            bandwidth_benchmark()
        else:
            print(NSYS_GUIDE)
        return

    if not args.model:
        parser.error("--tool profiler/all 需要 --model")

    if args.tool == "profiler":
        profile_engine(args.model, args.input_len, args.output_len,
                       args.num_prompts, args.trace_dir)
    else:
        bandwidth_benchmark()
        profile_engine(args.model, args.input_len, args.output_len,
                       args.num_prompts, args.trace_dir)
        print(NSYS_GUIDE)


if __name__ == "__main__":
    main()
