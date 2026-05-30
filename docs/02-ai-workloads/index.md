---
title: AI 计算工作负载基础
domain: ai-workloads
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# AI 计算工作负载基础

本目录只保留理解性能所必需的模型背景。重点不是解释模型任务效果，而是模型结构、输入形态和数值格式如何决定计算量、访存量、通信量和调度复杂度。

## 建议主题

- Transformer、Attention、MLP、Embedding、LayerNorm、MoE
- Prefill 与 Decode 的计算特征差异
- KV Cache 的显存占用、带宽压力和调度影响
- sequence length、batch size、concurrency 对延迟和吞吐的影响
- FP32、TF32、FP16、BF16、FP8、INT8、INT4
- 参数量、激活、optimizer state、checkpoint 对显存的影响
- arithmetic intensity、memory wall、compute-bound 与 memory-bound
- dense workload、sparse workload、dynamic shape 和 ragged batch

## 关键问题

- 一个模型层的 FLOPs 和内存访问量如何估算。
- Attention、MLP、Embedding、MoE 分别容易卡在哪类资源上。
- 长上下文为什么会改变 KV Cache、显存带宽和调度瓶颈。
- 量化、稀疏化、MoE、投机解码等技术如何改变 workload 形态。
- Workload 描述是否足以支持 Benchmark、容量估算和硬件选型。

## 建议记录字段

| 字段 | 说明 |
| --- | --- |
| workload | 模型结构、参数规模、层数、hidden size、head 数、MoE expert 数 |
| shape | batch size、sequence length、prefill/decode 比例、并发 |
| precision | 权重、激活、KV Cache、通信使用的数值格式 |
| bottleneck | compute、memory bandwidth、capacity、communication、scheduler |
| metrics | latency、throughput、tokens/s、MFU、HBM bandwidth、显存占用 |
