"""
tests/test_cudagraph_parity.py - CUDA Graph 与 Eager 输出一致性测试 (需 GPU)

对应 docs/01-cuda-graph.md 与 docs/06-debugging-stories.md。
这个测试防的是两个真实 bug:
1. decode context_lens 少算 1 (输出"慢一拍"重复)
2. kernel 用默认 stream launch, capture 漏录导致 replay 空转

运行: pytest tests/test_cudagraph_parity.py  (需 GPU + 本地模型)
CI 中跳过: pytest -m "not gpu"
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.gpu

MODEL_PATH = os.environ.get("NANO_VLLM_TEST_MODEL", "/root/autodl-tmp/models/Qwen3-0.6B")


@pytest.fixture(scope="module")
def engine_graph():
    torch = pytest.importorskip("torch")
    from nano_vllm import LLM
    llm = LLM(model=MODEL_PATH, max_model_len=2048,
              enable_fp8_kvcache=False, enforce_eager=False)
    yield llm
    del llm
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def engine_eager():
    # NCCL process group 不可重复初始化, eager 路径用子进程对拍脚本
    pytest.skip("eager 对比通过 test_gpu_smoke 的离线对拍完成")


def test_graph_generation_quality(engine_graph):
    """CUDA Graph 模式生成连贯文本（乱码/重复 = capture 或 context_len 回归）"""
    tok = engine_graph.tokenizer
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "用一句话说明什么是 KV Cache"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    from nano_vllm import SamplingParams
    out = engine_graph.generate([prompt],
                                SamplingParams(temperature=0.01, max_tokens=48))
    text = out[0]["text"]
    # 1) 非 token 级退化的启发式: 长度合理
    assert len(out[0]["token_ids"]) > 8, "生成过短, 可能 replay 空转"
    # 2) 无 U+FFFD 替换符串（detokenize 截断）
    assert "\ufffd" not in text, "输出含替换符, 增量解码回归"
    # 3) 无病态重复 (同一 4-gram 连续重复 > 6 次)
    words = text.split()
    if len(words) >= 12:
        for i in range(len(words) - 12):
            if words[i:i + 4] == words[i + 4:i + 8] == words[i + 8:i + 12]:
                pytest.fail(f"输出疑似退化为重复: {text[:80]!r}")


def test_graph_deterministic_with_temperature(engine_graph):
    """temperature 极低时, 同一 prompt 多次生成应基本一致（graph replay 正确性的弱断言）"""
    from nano_vllm import SamplingParams
    prompt = "数字 1, 2, 3,"
    sp = SamplingParams(temperature=1e-5, max_tokens=16)
    out1 = engine_graph.generate([prompt], sp)
    out2 = engine_graph.generate([prompt], sp)
    assert out1[0]["token_ids"] == out2[0]["token_ids"], \
        "低温度下输出不确定, graph replay 可能读到了脏数据"
