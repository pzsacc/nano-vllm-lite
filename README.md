# nano-vllm-lite
This repository is inspired by nano-vllm. Thanks! The core improvements include a CUDA-based Add+RMSNorm fused kernel, Chunked Prefill for a mixed scheduler, and FP8 KV Cache quantization achieved by rewriting the FlashAttention/PagedAttention Decode kernels with Triton. Hope this project helps you dive deeper into LLM inference mechanics! As follow is the main changements.

---

#### Chunked Prefill for a mixed scheduler

* `nanovllm/engine`
