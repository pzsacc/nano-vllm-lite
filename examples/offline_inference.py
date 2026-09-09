"""
examples/offline_inference.py - 离线批量推理示例

演示如何使用 nano-vllm 进行批量文本生成。

用法：
    python examples/offline_inference.py --model /path/to/Qwen3-0.6B
"""

import argparse
from nano_vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    # 初始化引擎
    llm = LLM(
        model=args.model,
        max_model_len=4096,
        enable_chunked_prefill=True,
        enable_fp8_kvcache=True,
    )

    # 准备 prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "What is the meaning of life?",
        "Write a short poem about the ocean:",
    ]

    # 采样参数
    sampling_params = SamplingParams(temperature=0.8, max_tokens=128)

    # 生成
    outputs = llm.generate(prompts, sampling_params)

    # 打印结果
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}")
        print(f"Output: {output['text']}")
        print(f"Tokens: {len(output['token_ids'])}")
        print("-" * 50)


if __name__ == "__main__":
    main()
