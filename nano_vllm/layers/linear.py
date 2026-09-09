"""
layers/linear.py - Tensor Parallel 线性层

实现多种并行策略的线性层，支持 NCCL 分布式训练/推理：
- ReplicatedLinear: 完全复制（无切分）
- ColumnParallelLinear: 按输出维度切分（每 rank 持有 output/tp_size 列）
- MergedColumnParallelLinear: 融合多个 ColumnParallel 投影（如 gate+up）
- QKVParallelLinear: 专门处理 Q/K/V 三路融合投影
- RowParallelLinear: 按输入维度切分（forward 后 all_reduce）

每个类自带 weight_loader 方法，配合 SafeTensors 加载器自动完成 TP 切分。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator: int, denominator: int) -> int:
    """整除辅助函数，断言可整除

    Args:
        numerator: 被除数
        denominator: 除数

    Returns:
        商
    """
    assert numerator % denominator == 0, f"{numerator} 不能被 {denominator} 整除"
    return numerator // denominator


class LinearBase(nn.Module):
    """并行线性层基类

    处理通用的 TP 初始化逻辑：获取 rank/world_size，
    创建 weight 参数并附加 weight_loader 属性。

    Args:
        input_size: 输入维度
        output_size: 输出维度（可能已经过 TP 切分）
        tp_dim: 权重切分的维度（0=列并行，1=行并行）
        bias: 是否使用偏置
    """

    def __init__(self, input_size: int, output_size: int, tp_dim: int = 0, bias: bool = False):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.bias = None

    def weight_loader(self, param, loaded_weight, *args):
        """权重加载（子类重写）"""
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """完全复制的线性层（所有 rank 持有相同权重）

    用于不需要并行的小型投影（如 LayerNorm 后的小 linear）。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, tp_dim=0, bias=bias)

    def weight_loader(self, param, loaded_weight):
        """直接复制完整权重"""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准线性变换"""
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """列并行线性层

    输出维度按 TP size 切分，每个 rank 持有 output_size/tp_size 列。
    前向传播无需通信（输入完整复制在每个 rank）。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        super().__init__(input_size, divide(output_size, tp_size), tp_dim=0, bias=bias)

    def weight_loader(self, param, loaded_weight):
        """从完整权重中切出当前 rank 的列分片"""
        shard_size = param.shape[self.tp_dim]
        loaded_weight = loaded_weight.narrow(self.tp_dim, self.tp_rank * shard_size, shard_size)
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """列并行前向（无通信）"""
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """融合列并行线性层

    将多个逻辑上独立的 ColumnParallel 投影融合为一个大矩阵乘法。
    典型用例：MLP 的 gate_proj + up_proj 融合为 gate_up_proj。

    Args:
        input_size: 输入维度
        output_sizes: 每个子投影的输出维度列表
        bias: 是否使用偏置
    """

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.output_sizes = output_sizes
        total_output = sum(output_sizes)
        super().__init__(input_size, total_output, bias=bias)

    def weight_loader(self, param, loaded_weight, loaded_shard_id: int):
        """按 shard_id 加载对应子投影的权重

        Args:
            param: 目标参数
            loaded_weight: 加载的权重
            loaded_shard_id: 子投影索引（如 0=gate, 1=up）
        """
        tp_size = self.tp_size
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
        shard_size = self.output_sizes[loaded_shard_id] // tp_size
        loaded_weight = loaded_weight.narrow(0, self.tp_rank * shard_size, shard_size)
        param.data.narrow(0, shard_offset, shard_size).copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """Q/K/V 融合投影的并行线性层

    将 Q、K、V 三个投影融合为一个矩阵乘法，支持 GQA（Q/K/V 头数不同）。
    输出布局：[Q_heads * head_dim | K_heads * head_dim | V_heads * head_dim]

    Args:
        hidden_size: 输入隐藏维度
        head_size: 每头维度
        total_num_heads: Q 的总头数
        total_num_kv_heads: KV 的总头数（GQA 时小于 Q）
        bias: 是否使用偏置
    """

    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int, bias: bool = False):
        self.head_size = head_size
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size * tp_size
        super().__init__(hidden_size, output_size, bias=bias)

    def weight_loader(self, param, loaded_weight, loaded_shard_id: str):
        """按 Q/K/V shard 加载权重

        Args:
            param: 目标参数
            loaded_weight: 加载的权重
            loaded_shard_id: "q", "k", 或 "v"
        """
        tp_size = self.tp_size
        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = self.num_heads * self.head_size
        elif loaded_shard_id == "k":
            shard_offset = self.num_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        elif loaded_shard_id == "v":
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        else:
            raise ValueError(f"未知 shard_id: {loaded_shard_id}")

        loaded_weight = loaded_weight.narrow(0, self.tp_rank * shard_size, shard_size)
        param.data.narrow(0, shard_offset, shard_size).copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """行并行线性层

    输入维度按 TP size 切分，每个 rank 持有 input_size/tp_size 行。
    前向传播后执行 all_reduce 聚合各 rank 的部分结果。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        super().__init__(divide(input_size, tp_size), output_size, tp_dim=1, bias=bias)

    def weight_loader(self, param, loaded_weight):
        """从完整权重中切出当前 rank 的行分片"""
        if param.dim() == 1:
            param.data.copy_(loaded_weight)
            return
        shard_size = param.shape[self.tp_dim]
        loaded_weight = loaded_weight.narrow(self.tp_dim, self.tp_rank * shard_size, shard_size)
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """行并行前向（包含 all_reduce 通信）"""
        y = F.linear(x, self.weight)
        if self.bias is not None and self.tp_rank == 0:
            y = y + self.bias
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
