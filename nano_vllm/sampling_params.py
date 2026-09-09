"""
sampling_params.py - 采样参数配置

定义 LLM 生成时的采样策略参数，包括温度、最大生成长度等。
当前仅支持基于温度的随机采样（Gumbel-max trick），不支持 greedy。
"""

from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    """采样参数

    Attributes:
        temperature: 采样温度，控制生成随机性（必须 > 1e-10）
        max_tokens: 单次请求最大生成 token 数
        ignore_eos: 是否忽略 EOS token 强制生成到 max_tokens
    """
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        """校验参数合法性"""
        assert self.temperature > 1e-10, (
            "temperature 必须 > 1e-10，当前不支持 greedy sampling"
        )
        assert self.max_tokens > 0, "max_tokens 必须为正整数"
