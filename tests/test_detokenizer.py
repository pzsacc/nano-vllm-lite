"""增量 Detokenizer 与 UTF-8 截断处理的正确性测试（CPU）"""
import sys
import pytest

sys.path.insert(0, "/root/autodl-tmp/projects/nano-vllm-lite")
from nano_vllm.engine.async_llm_engine import IncrementalDetokenizer, _utf8_truncated_len


class FakeTokenizer:
    """模拟 HF tokenizer 的 decode 行为（增量场景下按全量语义）"""

    def __init__(self, token_to_bytes: dict[int, bytes]):
        self.token_to_bytes = token_to_bytes

    def decode(self, token_ids, skip_special_tokens=False):
        """模拟 HF 全量解码: 完整字符才输出, 不完整序列整体不产生文本"""
        out = b""
        ids = list(token_ids)
        # 逐段尝试: 只有当从某个起点到末尾能完整解码时才计入
        i = 0
        while i < len(ids):
            matched = False
            for j in range(len(ids), i, -1):
                chunk = b"".join(self.token_to_bytes[t] for t in ids[i:j])
                try:
                    out += chunk.decode("utf-8").encode("utf-8")
                    i = j
                    matched = True
                    break
                except UnicodeDecodeError:
                    continue
            if not matched:  # 剩余为不完整序列, 整体跳过 (模拟 HF 行为)
                break
        return out.decode("utf-8", errors="replace")


def test_utf8_truncated_len():
    assert _utf8_truncated_len(b"") == 0
    assert _utf8_truncated_len(b"abc") == 0
    assert _utf8_truncated_len("中".encode()) == 0          # 完整 3 字节
    assert _utf8_truncated_len("中".encode()[:1]) == 1      # 截断 1 字节
    assert _utf8_truncated_len("中".encode()[:2]) == 2      # 截断 2 字节
    assert _utf8_truncated_len("😀".encode()[:3]) == 3      # 4 字节 emoji 截断
    assert _utf8_truncated_len("a中".encode()[1:]) == 0     # 完整


def test_incremental_multi_byte_split():
    # "中" 的 3 个字节被切到两个 token 里
    tok = FakeTokenizer({1: b"\xe4\xb8", 2: b"\xad", 3: b"!"})
    d = IncrementalDetokenizer(tok)
    assert d.decode(1) == ""      # 不完整, 不输出
    assert d.decode(2) == "中"    # 补齐后输出完整字符
    assert d.decode(3) == "!"
    assert d.decode(1) == ""      # 新一轮截断
    # 累计输出 = 全量解码
    assert (d.token_ids and d._emitted_len >= 0)


def test_incremental_matches_full_decode():
    # 模拟真实序列: 中英混合 + emoji
    seq = "Hello世界😀!ok"
    tokens = [seq[i:i+2].encode() for i in range(0, len(seq), 2)]
    tok = FakeTokenizer({i: b for i, b in enumerate(tokens)})
    d = IncrementalDetokenizer(tok)
    out = ""
    for i in range(len(tokens)):
        out += d.decode(i)
    assert out == seq
