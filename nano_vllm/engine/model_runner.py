"""
engine/model_runner.py - GPU 模型执行器

负责模型的 GPU 端执行，包括：
- KV Cache 显存分配（支持 FP16/FP8 两种精度）
- Prefill/Decode 输入准备（构造 attention mask、position、slot mapping）
- CUDA Graph 捕获与回放（加速 decode 阶段）
- Tensor Parallel 多进程协调（SharedMemory + NCCL）
- 模型 warmup（预热 torch.compile 和内存统计）
"""

import pickle
import torch
import torch.distributed as dist
from torch import inference_mode
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Event

from nano_vllm.config import Config
from nano_vllm.engine.sequence import Sequence
from nano_vllm.models.qwen3 import Qwen3ForCausalLM
from nano_vllm.layers.sampler import Sampler
from nano_vllm.utils.context import set_context, reset_context
from nano_vllm.utils.loader import load_model


class ModelRunner:
    """GPU 模型执行器

    管理模型推理的完整生命周期：初始化 → warmup → 分配 KV Cache →
    捕获 CUDA Graph → 执行推理循环。

    支持 Tensor Parallelism：rank 0 通过 SharedMemory 广播调用指令，
    所有 rank 通过 NCCL 同步计算结果。

    Args:
        config: 引擎配置
        rank: 当前进程的 TP rank
        events: 用于 TP 同步的 Event 列表（仅 rank 0 使用）
    """

    def __init__(self, config: Config, rank: int = 0, events: list[Event] = None):
        # 初始化分布式环境
        dist.init_process_group("nccl", f"tcp://localhost:2333",
                                world_size=config.tensor_parallel_size, rank=rank)
        torch.cuda.set_device(rank)
        self.rank = rank
        self.config = config
        self.events = events
        self.graphs = {}
        self.graph_pool = None

        # 加载模型
        prev_device = torch.get_default_device()
        prev_dtype = torch.get_default_dtype()
        torch.set_default_device("cuda")
        torch.set_default_dtype(config.hf_config.torch_dtype)
        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        torch.set_default_device(prev_device)
        torch.set_default_dtype(prev_dtype)

        # 预热、分配 KV Cache、捕获 CUDA Graph
        self.warmup_model()
        self.allocate_kv_cache()
        if not config.enforce_eager:
            self.capture_cudagraph()

        # TP 通信设置
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cpu")
        if rank == 0:
            self.shm = SharedMemory("nanovllm", create=True, size=2**20)
        else:
            self.shm = SharedMemory("nanovllm", create=False)
            self.loop()

    def exit(self):
        """清理资源：销毁 CUDA Graph、关闭共享内存、销毁进程组"""
        self.shm.close()
        if self.rank == 0:
            self.shm.unlink()
        if self.graphs:
            del self.graphs
        torch.cuda.synchronize()
        dist.destroy_process_group()

    # ======================== TP 通信 ========================

    def loop(self):
        """TP worker 主循环：等待 rank 0 广播指令并执行

        通过 SharedMemory + Event 实现零拷贝跨进程 RPC。
        """
        event = self.events
        while True:
            event.wait()
            event.clear()
            method, args = self.read_shm()
            if method == "exit":
                self.exit()
                return
            getattr(self, method)(*args)

    def read_shm(self):
        """从共享内存读取方法名和参数"""
        return pickle.loads(self.shm.buf[:])

    def write_shm(self, method: str, args: tuple):
        """向共享内存写入方法名和参数，并通知所有 worker"""
        data = pickle.dumps((method, args))
        self.shm.buf[:len(data)] = data
        for event in self.events:
            event.set()

    def call(self, method: str, *args):
        """TP 同步调用：广播指令后所有 rank 执行同一方法

        Args:
            method: 要调用的方法名
            *args: 方法参数
        """
        self.write_shm(method, args)
        return getattr(self, method)(*args)

    # ======================== 初始化阶段 ========================

    def warmup_model(self):
        """预热模型：触发 torch.compile JIT 编译，统计显存使用

        使用随机 dummy 输入执行一次完整的 prefill forward，
        让 PyTorch 完成所有 lazy 初始化和编译。
        """
        max_tokens = min(self.config.max_num_batched_tokens, self.config.max_model_len)
        dummy_seq = Sequence(list(range(max_tokens)))
        dummy_seq.block_table = list(range(dummy_seq.num_blocks))
        dummy_seq.num_scheduled_tokens = max_tokens
        self.run([dummy_seq], has_prefill=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def allocate_kv_cache(self):
        """根据剩余显存分配 KV Cache 物理块

        计算逻辑：
        1. 确定单个 block 的字节数（考虑 FP8 vs FP16）
        2. 用 (总显存 × 利用率 - 已用显存) / block字节数 得到可分配块数
        3. 创建全局 KV cache tensor 并分配给每个 Attention 层
        """
        hf_config = self.config.hf_config
        tp_size = dist.get_world_size()
        num_kv_heads = hf_config.num_key_value_heads // tp_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        num_layers = hf_config.num_hidden_layers
        block_size = self.config.kvcache_block_size

        # FP8: 1 byte/element; FP16: 2 bytes/element
        bytes_per_element = 1 if self.config.enable_fp8_kvcache else 2
        block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * bytes_per_element

        # 计算可分配的 block 数量
        total_mem = torch.cuda.get_device_properties(0).total_memory
        used_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        available = total_mem * self.config.gpu_memory_utilization - peak_mem
        self.config.num_kvcache_blocks = int(available) // block_bytes

        # 选择 KV cache 数据类型
        cache_dtype = torch.float8_e4m3fn if self.config.enable_fp8_kvcache else hf_config.torch_dtype

        # 分配全局 KV cache tensor: [2, layers, blocks, block_size, heads, head_dim]
        kv_cache = torch.zeros(
            2, num_layers, self.config.num_kvcache_blocks, block_size, num_kv_heads, head_dim,
            dtype=cache_dtype, device="cuda"
        )

        # 将 KV cache 切片分配给每个 Attention 模块
        from nano_vllm.layers.attention import Attention
        layer_idx = 0
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.k_cache = kv_cache[0, layer_idx]
                module.v_cache = kv_cache[1, layer_idx]
                layer_idx += 1

    # ======================== 输入准备 ========================

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        """将序列的 block table 对齐为统一长度的 tensor

        Args:
            seqs: 序列列表

        Returns:
            [batch_size, max_blocks] 的 int32 CUDA tensor，短序列用 -1 填充
        """
        max_blocks = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_blocks - len(seq.block_table))
            for seq in seqs
        ]
        return torch.tensor(block_tables, dtype=torch.int32, device="cuda")

    def prepare_prefill(self, seqs: list[Sequence]):
        """准备 Prefill 阶段的输入 tensor

        构造 flash_attn_varlen 所需的 cu_seqlens、positions、slot_mapping。
        支持 prefix cache 命中时的 partial prefill。

        Args:
            seqs: 本步被调度的 prefill 序列
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping = []

        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            input_ids.extend(seq.token_ids[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seq.num_scheduled_tokens)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)

            for pos in range(start, end):
                block_idx = pos // self.config.kvcache_block_size
                block_offset = pos % self.config.kvcache_block_size
                slot = seq.block_table[block_idx] * self.config.kvcache_block_size + block_offset
                slot_mapping.append(slot)

        self.input_ids = torch.tensor(input_ids, dtype=torch.long, device="cuda")
        self.positions = torch.tensor(positions, dtype=torch.long, device="cuda")

        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cuda")
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device="cuda")

        block_tables = self.prepare_block_tables(seqs) if cu_seqlens_k[-1] > cu_seqlens_q[-1] else None

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(s.num_scheduled_tokens for s in seqs),
            max_seqlen_k=max(s.num_cached_tokens + s.num_scheduled_tokens for s in seqs),
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )

    def prepare_decode(self, seqs: list[Sequence]):
        """准备 Decode 阶段的输入 tensor

        每个序列只处理最后一个 token，构造 paged attention 所需元数据。
        额外保存 context tensor 供 CUDA Graph replay 时拷贝使用。

        Args:
            seqs: 本步被调度的 decode 序列
        """
        input_ids = []
        positions = []
        context_lens = []
        slot_mapping = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(seq.num_tokens - 1)
            context_lens.append(seq.num_tokens - 1)
            last_block_idx = len(seq.block_table) - 1
            last_offset = (seq.num_tokens - 1) % self.config.kvcache_block_size
            slot = seq.block_table[last_block_idx] * self.config.kvcache_block_size + last_offset
            slot_mapping.append(slot)

        self.input_ids = torch.tensor(input_ids, dtype=torch.long, device="cuda")
        self.positions = torch.tensor(positions, dtype=torch.long, device="cuda")
        self.decode_slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device="cuda")
        self.decode_context_lens = torch.tensor(context_lens, dtype=torch.int32, device="cuda")
        self.decode_block_tables = self.prepare_block_tables(seqs)

        set_context(
            is_prefill=False,
            slot_mapping=self.decode_slot_mapping,
            context_lens=self.decode_context_lens,
            block_tables=self.decode_block_tables,
        )

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        """准备采样所需的温度参数

        Args:
            seqs: 序列列表

        Returns:
            [batch_size] 的温度 tensor
        """
        return torch.tensor([seq.temperature for seq in seqs], device="cuda")

    # ======================== 推理执行 ========================

    @inference_mode()
    def run_model(self, has_prefill: bool) -> torch.Tensor:
        """执行模型 forward pass

        根据当前状态选择执行路径：
        - Prefill / enforce_eager / 大 batch：直接 eager 执行
        - Decode 小 batch：回放预捕获的 CUDA Graph

        Args:
            has_prefill: 当前 batch 是否包含 prefill 序列

        Returns:
            logits tensor [num_tokens_to_sample, vocab_size]
        """
        batch_size = self.input_ids.shape[0]

        # Prefill 或 eager 模式：直接执行
        if has_prefill or self.config.enforce_eager or batch_size > 512:
            hidden = self.model(self.input_ids, self.positions)
            return self.model.compute_logits(hidden)

        # Decode: 选择最小满足 batch_size 的 CUDA Graph
        graph_bs = batch_size
        if graph_bs not in self.graphs:
            for candidate in sorted(self.graphs.keys()):
                if candidate >= batch_size:
                    graph_bs = candidate
                    break

        # 将输入拷贝到 graph 的 static tensor 中（地址固定，graph 可识别）
        gv = self.graph_vars[graph_bs]
        gv["input_ids"][:batch_size].copy_(self.input_ids)
        gv["positions"][:batch_size].copy_(self.positions)
        gv["slot_mapping"][:batch_size].copy_(self.decode_slot_mapping)
        gv["context_lens"][:batch_size].copy_(self.decode_context_lens)
        bt = self.decode_block_tables
        gv["block_tables"][:batch_size, :bt.shape[1]].copy_(bt)

        # 更新全局 context 指向 graph 的 static tensor（保持地址不变）
        set_context(
            is_prefill=False,
            slot_mapping=gv["slot_mapping"],
            context_lens=gv["context_lens"],
            block_tables=gv["block_tables"],
        )

        # 回放 CUDA Graph
        self.graphs[graph_bs].replay()

        # 截取实际 batch 的输出
        logits = self.model.compute_logits(self.graph_hidden[graph_bs][:batch_size])
        return logits

    def run(self, seqs: list[Sequence], has_prefill: bool) -> list[int] | None:
        """完整的单步推理：准备输入 → forward → 采样

        Args:
            seqs: 本步被调度的序列
            has_prefill: 是否包含 prefill 序列

        Returns:
            rank 0 返回采样的 token ID 列表，其他 rank 返回 None
        """
        if has_prefill:
            self.prepare_prefill(seqs)
        else:
            self.prepare_decode(seqs)

        logits = self.run_model(has_prefill)

        # 只有 rank 0 执行采样
        if self.rank == 0:
            temperatures = self.prepare_sample(seqs)
            token_ids = self.sampler(logits, temperatures)
            reset_context()
            return token_ids.tolist()
        else:
            reset_context()
            return None

    # ======================== CUDA Graph ========================

    def capture_cudagraph(self):
        """捕获 Decode 阶段的 CUDA Graph

        核心设计：为每个 batch_size 预分配一组 static tensor（固定内存地址），
        graph 捕获时使用这些 static tensor，replay 前将真实数据拷贝进去。
        这保证 graph replay 时读写的地址与 capture 时一致。
        """
        max_bs = min(self.config.max_num_seqs, 512)
        batch_sizes = []
        bs = 1
        while bs <= max_bs:
            batch_sizes.append(bs)
            bs *= 2
        if batch_sizes[-1] != max_bs:
            batch_sizes.append(max_bs)

        self.graph_pool = torch.cuda.graph_pool_handle()
        self.graph_vars = {}
        self.graph_hidden = {}
        max_blocks_per_seq = (self.config.max_model_len + self.config.kvcache_block_size - 1) // self.config.kvcache_block_size

        # 为每个 batch_size 预分配 static context tensor
        for graph_bs in batch_sizes:
            self.graph_vars[graph_bs] = {
                "input_ids": torch.zeros(graph_bs, dtype=torch.long, device="cuda"),
                "positions": torch.zeros(graph_bs, dtype=torch.long, device="cuda"),
                "slot_mapping": torch.zeros(graph_bs, dtype=torch.int32, device="cuda"),
                "context_lens": torch.ones(graph_bs, dtype=torch.int32, device="cuda"),
                "block_tables": torch.zeros(graph_bs, max_blocks_per_seq, dtype=torch.int32, device="cuda"),
            }

        # 从大到小捕获（共享 memory pool）
        for graph_bs in reversed(batch_sizes):
            gv = self.graph_vars[graph_bs]

            # 设置 static tensor 作为模型输入
            self.input_ids = gv["input_ids"]
            self.positions = gv["positions"]
            set_context(
                is_prefill=False,
                slot_mapping=gv["slot_mapping"],
                context_lens=gv["context_lens"],
                block_tables=gv["block_tables"],
            )

            # Warmup run（让 PyTorch 分配所有中间 buffer）
            hidden = self.model(self.input_ids, self.positions)
            reset_context()

            # 正式 Capture
            set_context(
                is_prefill=False,
                slot_mapping=gv["slot_mapping"],
                context_lens=gv["context_lens"],
                block_tables=gv["block_tables"],
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                self.graph_hidden[graph_bs] = self.model(self.input_ids, self.positions)
            reset_context()
            self.graphs[graph_bs] = graph
