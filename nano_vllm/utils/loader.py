"""
utils/loader.py - 模型权重加载器

从 SafeTensors 格式的权重文件加载模型参数，支持：
- 自动按 packed_modules_mapping 拆分融合权重（如 qkv_proj → q/k/v）
- 每个 Linear 层自带 weight_loader 方法处理 TP 切分
"""

from glob import glob
from safetensors.torch import load_file


def default_weight_loader(param, loaded_weight):
    """默认权重加载：直接复制

    Args:
        param: 目标参数 tensor
        loaded_weight: 从文件加载的权重 tensor
    """
    param.data.copy_(loaded_weight)


def load_model(model, path: str):
    """加载模型权重到 model 实例

    支持融合权重映射：通过 model.packed_modules_mapping 字典，
    将原始 HuggingFace checkpoint 中分散的 q_proj/k_proj/v_proj
    映射到融合后的 qkv_proj 等参数上。

    Args:
        model: nn.Module 实例，需定义 packed_modules_mapping 属性
        path: 权重文件目录路径，包含 *.safetensors 文件
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    state_dict = dict(model.named_parameters())

    for weight_file in sorted(glob(f"{path}/*.safetensors")):
        weights = load_file(weight_file)
        for name, tensor in weights.items():
            # 检查是否命中融合权重映射规则
            mapped = False
            for original_key, (fused_key, shard_id) in packed_modules_mapping.items():
                if original_key in name:
                    param_name = name.replace(original_key, fused_key)
                    if param_name in state_dict:
                        param = state_dict[param_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, tensor, shard_id)
                        mapped = True
                    break

            if not mapped:
                if name in state_dict:
                    param = state_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, tensor)
