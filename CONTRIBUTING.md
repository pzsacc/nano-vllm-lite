# Contributing to nano-vllm-lite

感谢关注！本项目目标是"小而精"的推理引擎教学项目，PR 请守住这个边界。

## 环境搭建

```bash
git clone https://github.com/pzsacc/nano-vllm-lite.git
cd nano-vllm-lite
pip install -e . --no-build-isolation   # 含 CUDA kernel 编译
pytest tests/ -m "not gpu"              # 快速验证
pytest tests/                           # GPU 环境全量
```

模型权重：Qwen3-0.6B（HuggingFace），通过
`NANO_VLLM_TEST_MODEL=/path/to/model` 指定给 GPU 测试。

## PR 规范

1. **性能改动必须附带测量**：用 `benchmarks/` 三层体系跑出前后对比（贴数字或 JSON）
2. **行为改动必须附测试**：修 bug 请先写一个能复现问题的失败测试
3. **文档同步**：新优化 → 新增 `docs/0X-*.md`（结构：现象→根因→实现→代价→教学点）；
   改动已有行为 → 更新对应篇章
4. 单 commit 单主题；中文 commit message 优先（与仓库历史一致）

## 好的第一个 Issue（见 Issue 区）

- async 引擎空闲轮询改条件变量
- 流式输出过滤 EOS 特殊 token
- mixed 压测场景的到达分布参数化

## 不接受的方向（保持"精"）

- 多模型支持的重构（模型注册已在 roadmap，请先开 issue 讨论）
- 通用 vLLM 特性搬运（我们只要教学价值高、代码量小的子集）
