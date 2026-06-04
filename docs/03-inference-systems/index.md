---
title: 推理系统与优化
domain: inference-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 推理系统与优化

本目录关注模型进入在线服务后的系统问题：如何降低延迟、提高吞吐、控制显存、隔离租户、处理长尾请求，并让服务行为可测、可复现。

## 建议主题

- Prefill、Decode、streaming output、request lifecycle
- KV Cache、PagedAttention、prefix cache、cache eviction
- continuous batching、dynamic batching、admission control
- speculative decoding、early exit、parallel decoding
- quantization、weight-only quantization、KV Cache quantization
- vLLM、TensorRT-LLM、TGI、llama.cpp、SGLang
- routing、rate limit、queueing、backpressure、fallback
- TTFT、TPOT、p50/p95/p99 latency、tokens/s、GPU memory

## 关键问题

- 当前目标是降低单请求延迟、提升整体吞吐，还是控制尾延迟。
- 请求队列、batching、KV Cache、kernel 执行和网络返回各占多少时间。
- 长上下文和高并发下显存容量、带宽和碎片化如何变化。
- 优化是否影响输出质量、稳定性、隔离性或可观测性。
- Benchmark 是否区分 prefill-heavy、decode-heavy 和 mixed workload。
