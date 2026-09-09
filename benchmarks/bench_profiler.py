"""
benchmarks/bench_profiler.py - PyTorch/NVIDIA 官方 Profiler 压测

使用 torch.profiler (集成 NVIDIA CUPTI) 对 nano-vllm 推理引擎进行详细性能分析，
生成可在 TensorBoard 或 Chrome Trace Viewer 中查看的 profile 结果。

功能:
- 分别 profile Prefill 和 Decode 阶段
- 记录 CPU/CUDA 算子耗时、GPU kernel 耗时、显存使用
- 导出 Chrome Trace JSON + TensorBoard trace
- 可选: 使用 NVIDIA Nsight Systems 的标记 (nvtx)

用法:
    python -m benchmarks.bench_profiler --model /path/to/Qwen3-0.6B
    python -m benchmarks.bench_profiler --model /path/to/Qwen3-0.6B --export-trace ./traces
    python -m benchmarks.bench_profiler --model /path/to/Qwen3-0.6B --num-prompts 16 --input-len 256 --output-len 64

查看结果:
    # Chrome Trace: 在 Chrome 浏览器打开 chrome://tracing 加载 .json 文件
    # TensorBoard:  tensorboard --logdir=./profiler_traces
"""

import argparse
import os
import time

import torch
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

from nano_vllm import LLM, SamplingParams


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_warmup(llm: LLM, input_len: int, output_len: int):
    """预热引擎，确保 torch.compile/CUDA Graph 已完成"""
    print("[Warmup] 执行预热推理...")
    prompt = list(range(input_len))
    sp = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)
    llm.generate([prompt], sp)
    torch.cuda.synchronize()
    print("[Warmup] 完成\n")


def profile_generation(llm: LLM, prompts: list, sampling_params: SamplingParams,
                       trace_dir: str, num_steps: int = None):
    """使用 torch.profiler 对完整生成过程进行 profile

    Args:
        llm: 引擎实例
        prompts: prompt 列表
        sampling_params: 采样参数
        trace_dir: trace 输出目录
        num_steps: 限制 profile 的 step 数量 (None 则运行到结束)
    """
    for prompt in prompts:
        llm.add_request(prompt, sampling_params)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    step_count = 0
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=tensorboard_trace_handler(trace_dir),
    ) as prof:
        while not llm.is_finished():
            if num_steps and step_count >= num_steps:
                break

            nvtx.range_push(f"step_{step_count}")
            with record_function(f"engine_step_{step_count}"):
                llm.step()
            nvtx.range_pop()

            prof.step()
            step_count += 1

    torch.cuda.synchronize()
    return prof, step_count


def profile_prefill_only(llm: LLM, input_len: int, trace_dir: str, num_prompts: int = 4):
    """单独 profile Prefill 阶段"""
    print_separator("Profiling Prefill Phase")

    prompts = [list(range(input_len)) for _ in range(num_prompts)]
    sp = SamplingParams(temperature=1.0, max_tokens=1, ignore_eos=True)

    prof, steps = profile_generation(llm, prompts, sp, os.path.join(trace_dir, "prefill"))

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print(f"\n[Prefill] 完成 {steps} steps, {num_prompts} prompts × {input_len} tokens")

    return prof


def profile_decode_only(llm: LLM, input_len: int, output_len: int,
                        trace_dir: str, num_prompts: int = 4):
    """Profile Decode 阶段 (先 prefill 完再 profile decode 部分)"""
    print_separator("Profiling Decode Phase")

    prompts = [list(range(input_len)) for _ in range(num_prompts)]
    sp = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)

    for prompt in prompts:
        llm.add_request(prompt, sp)

    # 先完成 prefill (不 profile)
    prefill_steps = 0
    while not llm.is_finished():
        llm.step()
        prefill_steps += 1
        if prefill_steps >= num_prompts:
            break

    torch.cuda.synchronize()

    # 现在 profile decode 阶段
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    decode_dir = os.path.join(trace_dir, "decode")

    step_count = 0
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=tensorboard_trace_handler(decode_dir),
    ) as prof:
        while not llm.is_finished():
            nvtx.range_push(f"decode_step_{step_count}")
            with record_function(f"decode_step_{step_count}"):
                llm.step()
            nvtx.range_pop()

            prof.step()
            step_count += 1

    torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print(f"\n[Decode] 完成 {step_count} decode steps")

    return prof


def profile_full_e2e(llm: LLM, input_len: int, output_len: int,
                     trace_dir: str, num_prompts: int = 4):
    """端到端 profile: Prefill + Decode 全流程"""
    print_separator("Profiling End-to-End (Prefill + Decode)")

    prompts = [list(range(input_len)) for _ in range(num_prompts)]
    sp = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)

    start = time.perf_counter()
    prof, steps = profile_generation(llm, prompts, sp, os.path.join(trace_dir, "e2e"))
    elapsed = time.perf_counter() - start

    total_output_tokens = num_prompts * output_len
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    print(f"\n[E2E] {num_prompts} prompts × (in={input_len}, out={output_len})")
    print(f"[E2E] 总耗时: {elapsed:.3f}s, 总 steps: {steps}")
    print(f"[E2E] 吞吐: {total_output_tokens / elapsed:.1f} tokens/s")

    return prof


def print_gpu_memory_summary():
    """打印 GPU 显存使用摘要"""
    print_separator("GPU Memory Summary")
    print(torch.cuda.memory_summary(abbreviated=True))


def export_chrome_trace(prof, path: str):
    """导出 Chrome Trace JSON 文件"""
    prof.export_chrome_trace(path)
    print(f"[Export] Chrome Trace 已导出: {path}")


def print_kernel_summary(prof):
    """打印 CUDA Kernel 统计"""
    print_separator("Top CUDA Kernels (by total time)")
    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="self_cuda_time_total", row_limit=15
    ))


def main():
    parser = argparse.ArgumentParser(description="nano-vllm PyTorch/NVIDIA Profiler 压测")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--input-len", type=int, default=512, help="输入序列长度")
    parser.add_argument("--output-len", type=int, default=128, help="输出序列长度")
    parser.add_argument("--num-prompts", type=int, default=4, help="并发请求数")
    parser.add_argument("--export-trace", type=str, default="./profiler_traces",
                        help="Trace 输出目录")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["prefill", "decode", "e2e", "all"],
                        help="Profile 模式: prefill/decode/e2e/all")
    parser.add_argument("--max-model-len", type=int, default=4096, help="最大模型长度")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="禁用 CUDA Graph (方便 profile 细节)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="GPU 显存利用率 (为 profiler 预留空间，默认 0.85)")
    args = parser.parse_args()

    os.makedirs(args.export_trace, exist_ok=True)

    print_separator("nano-vllm Profiler Benchmark")
    print(f"Model:       {args.model}")
    print(f"Input len:   {args.input_len}")
    print(f"Output len:  {args.output_len}")
    print(f"Num prompts: {args.num_prompts}")
    print(f"Mode:        {args.mode}")
    print(f"Eager mode:  {args.enforce_eager}")
    print(f"Trace dir:   {args.export_trace}")
    print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(f"CUDA:        {torch.version.cuda}")
    print(f"PyTorch:     {torch.__version__}")

    # 初始化引擎
    print("\n[Init] 正在初始化引擎...")
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        enable_chunked_prefill=True,
        enable_fp8_kvcache=True,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print("[Init] 引擎初始化完成")

    # 预热
    run_warmup(llm, args.input_len, args.output_len)

    # Profile
    if args.mode in ("prefill", "all"):
        prof_prefill = profile_prefill_only(
            llm, args.input_len, args.export_trace, args.num_prompts
        )
        export_chrome_trace(
            prof_prefill,
            os.path.join(args.export_trace, "prefill_trace.json")
        )

    if args.mode in ("decode", "all"):
        prof_decode = profile_decode_only(
            llm, args.input_len, args.output_len, args.export_trace, args.num_prompts
        )
        export_chrome_trace(
            prof_decode,
            os.path.join(args.export_trace, "decode_trace.json")
        )

    if args.mode in ("e2e", "all"):
        prof_e2e = profile_full_e2e(
            llm, args.input_len, args.output_len, args.export_trace, args.num_prompts
        )
        export_chrome_trace(
            prof_e2e,
            os.path.join(args.export_trace, "e2e_trace.json")
        )
        print_kernel_summary(prof_e2e)

    # 显存摘要
    print_gpu_memory_summary()

    print_separator("Profile 完成")
    print(f"Trace 文件位于: {args.export_trace}/")
    print("查看方式:")
    print("  - Chrome Trace:  chrome://tracing → Load *.json")
    print("  - TensorBoard:   tensorboard --logdir={args.export_trace}")
    print("  - Nsight Systems: nsys profile python -m benchmarks.bench_profiler --model ... --enforce-eager")


if __name__ == "__main__":
    main()
