# 08 · Async 引擎 — 流式输出的工程细节

[上一篇](07-benchmarks-5090.md) | [目录](README.md)

> 核心代码: `engine/async_llm_engine.py` · 正确性测试: `tests/test_detokenizer.py`

## 架构

```
┌─ asyncio 事件循环 (HTTP 进程) ──────────────────────┐
│  generate_stream() async generator                  │
│    │ 提交 (prompt, sp, queue)                        │
│    ▼                                                │
│  new_requests 列表 ←──── call_soon_threadsafe ────┐ │
│  asyncio.Queue (每请求一个) ←─────────────────────┤ │
└──────────────────────────────────────────────────┼─┘
┌─ 引擎线程 ───────────────────────────────────────┴─┐
│  _background_loop:                                  │
│    1. 处理 abort_requests (客户端取消)               │
│    2. 收 new_requests → scheduler.add               │
│    3. engine.step_stream() → 每 token 投递到 queue  │
└─────────────────────────────────────────────────────┘
```

设计原则：**同步引擎一行不改**，async 是外壳。引擎线程照常跑
`schedule → run → postprocess`，跨线程只传 `(text, is_finished)` 元组。

## 三个踩过坑的工程细节

### 1. 客户端取消必须传播到引擎

async generator 被 `aclose()`/断连后，若无取消机制，引擎会**把整条序列
生成到 EOS**，token 堆进无人消费的队列——算力浪费 + 内存泄漏。

```python
except asyncio.CancelledError:
    seq_id = self._find_seq_id(stream_queue)
    if seq_id is not None:
        self.abort_requests.add(seq_id)   # 引擎线程下一轮消费
    raise
```

配套的 `scheduler.abort(seq_id)` 从 waiting/running 移除序列并释放 KV block。
验证方法：生成 5 个 token 后 cancel，1 秒后断言引擎队列长度为 0。

### 2. 逐 token decode 会打碎多字节字符

`tokenizer.decode([token_id])` 在"中"/"😀"这类跨 token 的 UTF-8 序列上
会输出 `\ufffd` 替换符。解法是增量缓冲：

```python
class IncrementalDetokenizer:
    def decode(self, new_token_id):
        # 全量解码 → 只 flush 完整的字符增量
        delta = cur_bytes[self._emitted_len:]
        flush_len = len(delta) - _utf8_truncated_len(delta)
        self._emitted_len += flush_len
        return delta[:flush_len].decode(...)
```

`_utf8_truncated_len` 从尾部回扫 UTF-8 续字节（`10xxxxxx`），判断尾部
是否是被截断的多字节字符前缀。测试：把"中"的三个字节切到两个 token，
断言第一个 token 输出为空、第二个输出完整的"中"。

### 3. `asyncio.get_event_loop()` 的坑

Python 3.12 起在无运行 loop 的上下文里调用会告警/报错。正确做法是
延迟到第一次 `generate_stream` 时用 `asyncio.get_running_loop()`。

## 已知局限（也是读者的练习题）

| 局限 | 为什么能跑 | 更好的做法 |
|------|-----------|-----------|
| 空闲轮询 `time.sleep(0.001)` | 1ms 粒度下 CPU 开销可接受 | `threading.Event` 条件变量唤醒 |
| `new_requests` 跨线程无锁 | CPython GIL 保证 list append/pop 原子 | `queue.Queue` 或锁 |
| EOS token 文本仍会 yield | skip_special_tokens=False | 流式过滤需增量判断 |

## 扩展方向：真实生产架构长什么样

本项目单进程方案是教学起点。生产系统（vLLM V1 / SGLang 共同的形态）：

- **进程边界**：调度器+KV 管理跑在独立进程（vLLM 叫 EngineCore），HTTP 前端
  asyncio 进程通过 **ZMQ** 与其通信——前端永远不会被引擎阻塞
- **socket 选型**：多客户端→核心用 DEALER/ROUTER（请求排队），核心→前端用
  PUSH/PULL（天然负载均衡 + 高水位背压）
- **流量不对称**：请求可以远超引擎处理速度（前端只管收），输出侧用
  "覆盖式聚合"防止慢消费者拖垮引擎
- **SGLang 特有**：RadixAttention 把前缀缓存组织成 radix tree（比本项目
  块级哈希更细粒度），tokenizer/detokenizer 也拆成独立进程

建议路线：读懂本项目 async 引擎后，读 vLLM 的 `vllm/v1/engine/core.py`
（EngineCore 进程）与 `core_client.py`（ZMQ 客户端族），对照理解为什么要
拆进程。

---

## 本篇速记

- 一句话：**同步引擎 + 后台线程 + 每请求队列 = 最小可用流式方案**
- 三个坑：取消传播 / UTF-8 截断 / loop 获取
- 相关测试：`tests/test_detokenizer.py`（CPU 可跑）

[上一篇](07-benchmarks-5090.md) | [目录](README.md)
