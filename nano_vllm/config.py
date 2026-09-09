"""
config.py - 引擎全局配置模块

定义推理引擎的所有可调参数，包括模型路径、批处理限制、内存管理、
并行策略和特性开关（Feature Flags）。通过 dataclass 提供类型安全的配置。
"""

import os
from dataclasses import dataclass, field
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    """推理引擎全局配置

    Attributes:
        model: 模型权重目录路径（HuggingFace 格式）
        max_num_batched_tokens: 单步最大处理 token 数
        max_num_seqs: 单步最大并发序列数
        max_model_len: 模型支持的最大上下文长度
        gpu_memory_utilization: GPU 显存利用率上限
        tensor_parallel_size: 张量并行 world size
        enforce_eager: 强制使用 eager 模式（禁用 CUDA Graph）
        kvcache_block_size: KV Cache 页块大小（token 数）
        chunk_size: Chunked Prefill 分块大小
        enable_chunked_prefill: 是否启用 Chunked Prefill 混合调度
        enable_fp8_kvcache: 是否启用 FP8 KV Cache 量化
        enable_prefix_caching: 是否启用前缀缓存复用
    """

    # ======================== 基础配置 ========================
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False

    # ======================== KV Cache 配置 ========================
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # ======================== Feature Flags ========================
    chunk_size: int = 1024
    enable_chunked_prefill: bool = True
    enable_fp8_kvcache: bool = True
    enable_prefix_caching: bool = True

    # ======================== 内部状态（自动填充） ========================
    hf_config: AutoConfig | None = field(default=None, repr=False)
    eos: int = -1

    def __post_init__(self):
        """验证配置合法性并加载 HuggingFace 模型配置"""
        assert os.path.isdir(self.model), f"模型路径不存在: {self.model}"
        assert self.kvcache_block_size % 256 == 0, "block_size 必须是 256 的倍数"
        assert 1 <= self.tensor_parallel_size <= 8, "TP size 必须在 [1, 8] 范围内"
        assert 0.0 < self.gpu_memory_utilization <= 1.0, "显存利用率必须在 (0, 1] 范围内"

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
