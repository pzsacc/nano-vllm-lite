"""KV Cache 容量探针 — FP16 vs FP8（5090 32GB 实测）

不跑请求，直接实例化引擎读 num_kvcache_blocks，
换算 max_model_len=4096 下的最大满长度并发 seqs。
"""
import sys
import torch
from nano_vllm import LLM
from nano_vllm.config import Config

MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"
MAX_LEN = 4096

label = sys.argv[1] if len(sys.argv) > 1 else "FP16"
fp8 = label.upper() == "FP8"

cfg_kw = dict(max_model_len=MAX_LEN, enable_fp8_kvcache=fp8)
engine = LLM(MODEL, **cfg_kw)
blocks = engine.config.num_kvcache_blocks
hf = engine.config.hf_config
bytes_per_token = (hf.num_hidden_layers * 2
                   * (hf.num_key_value_heads * hf.head_dim)
                   * (1 if fp8 else 2))
seqs = blocks * engine.config.kvcache_block_size // MAX_LEN
used_gb = blocks * engine.config.kvcache_block_size * bytes_per_token / 2**30

free, total = torch.cuda.mem_get_info()
print(f"\n===== {label} =====")
print(f"KV 块数: {blocks} (block_size={engine.config.kvcache_block_size})")
print(f"KV 池: {used_gb:.1f} GiB | 每 token {bytes_per_token} B")
print(f"满 4096 上下文最大并发: {seqs} seqs")
torch.cuda.empty_cache()
