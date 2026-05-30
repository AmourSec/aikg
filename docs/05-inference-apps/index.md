---
title: 推理算法与高效服务
domain: inference-apps
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 推理算法与高效服务

本目录关注模型推理阶段的算法、系统实现和性能分析，重点是理解高效推理论文和系统实现如何被复现、比较和扩展。

## 建议主题

- 在线推理、离线批处理、流式输出
- KV Cache、PagedAttention、Continuous Batching
- Speculative Decoding、Early Exit、MoE Routing
- 量化、剪枝、蒸馏、权重共享
- Serving Runtime、调度、缓存、隔离和容错
- Latency、Throughput、TTFT、TPOT、显存占用

## 关键问题

- 推理瓶颈来自计算、显存、内存带宽还是通信
- Batch size、上下文长度和并发如何影响性能
- 论文中的吞吐和延迟指标是否可复现
- Serving 系统如何处理长尾请求、失败和资源隔离
- 模型压缩是否改变质量、鲁棒性和校准性
