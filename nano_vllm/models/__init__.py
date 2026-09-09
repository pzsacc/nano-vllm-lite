"""
models/__init__.py - 模型注册表

提供模型架构的注册和自动发现机制。
通过 HuggingFace AutoConfig 的 model_type 字段自动匹配对应的模型类。
"""

from nano_vllm.models.qwen3 import Qwen3ForCausalLM

# 模型注册表：model_type → model_class
MODEL_REGISTRY = {
    "qwen3": Qwen3ForCausalLM,
}


def get_model_class(model_type: str):
    """根据 model_type 获取对应的模型类

    Args:
        model_type: HuggingFace config 中的 model_type 字段

    Returns:
        模型类

    Raises:
        ValueError: 不支持的模型类型
    """
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"不支持的模型类型 '{model_type}'，当前支持: {supported}")
    return MODEL_REGISTRY[model_type]
