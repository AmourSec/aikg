---
title: 入门导读
domain: getting-started
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 入门导读

本目录面向 AI Infra / AI Systems / Efficient AI Computing 方向的新生。入门目标不是系统学习所有模型算法，而是建立一个性能视角：一个 AI workload 为什么慢、为什么贵、为什么不稳定，以及可以从哪些层面改进。

## 建议先建立的共同语言

- Workload、batch shape、sequence length、precision、concurrency
- Latency、throughput、TTFT、TPOT、tail latency
- FLOPs、memory bandwidth、HBM、cache、communication
- GPU utilization、MFU、occupancy、kernel time、queueing delay
- Prefill、decode、KV Cache、activation、optimizer state
- Profiling、Benchmark、Roofline、capacity planning

## 入门路径

1. 用 `AI 基础概念` 建立 token、tensor、parameter、loss、gradient、optimizer 的共同语言。
2. 用 `Transformer 流程与原理` 理解 Attention、MLP、LayerNorm、Logits 的基本数据流。
3. 用 `训练过程与原理` 和 `推理过程与原理` 区分 forward/backward、prefill/decode、KV Cache 等关键过程。
4. 用 `AI 计算工作负载基础` 理解性能瓶颈来自哪里。
5. 用 `推理系统与服务优化` 理解在线服务的延迟、吞吐和调度。
6. 用 `训练系统与分布式计算` 理解多卡、多机训练的并行和通信。
7. 用 `Kernel、算子与编译优化` 理解单个算子如何被做快。
8. 用 `AI 加速器与计算架构` 理解硬件如何约束上层系统。
9. 用 `性能分析、Benchmark 与容量建模` 验证任何优化是否真实有效。

## 写作要求

- 不写泛泛的模型效果提升，只写与性能、效率、稳定性、复现有关的结论。
- 所有性能结论都要写清楚 workload、硬件、软件版本、输入 shape、并发模型和指标口径。
- 技术比较必须说明瓶颈假设，例如 compute-bound、memory-bound、communication-bound 或 scheduler-bound。
- 论文笔记必须把核心方法映射到系统层级：workload、runtime、kernel、accelerator、cluster、measurement。

## 关键问题

- 这个问题的主要瓶颈是在计算、显存、带宽、通信、调度还是数据路径。
- 指标是服务端吞吐、单请求延迟、尾延迟、训练 step time、扩展效率还是能效。
- 优化是否改变了质量、稳定性、隔离性或复现性。
- 结论是否能被后续实验、AI 检索或系统设计复用。
