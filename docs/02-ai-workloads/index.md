---
title: AI 计算工作负载基础
domain: ai-workloads
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# AI 计算工作负载基础

本目录是 AI Infra 的基础课。它先解释模型、Transformer、训练和推理的基本流程，再把这些流程翻译成计算量、访存量、显存占用、通信量和调度复杂度。

## 入门顺序

1. [AI 基础概念](ai-fundamentals.md)：理解模型、参数、张量、token、loss、训练和推理这些共同语言。
2. [Transformer 流程与原理](transformer.md)：理解 Embedding、Attention、MLP、残差、LayerNorm、Logits 的数据流。
3. [训练过程与原理](training-primer.md)：理解 forward、loss、backward、optimizer、checkpoint 为什么消耗计算和显存。
4. [推理过程与原理](inference-primer.md)：理解 prefill、decode、KV Cache、sampling、batching 为什么决定延迟和吞吐。
5. [数据与输入路径](data-paths.md)：理解数据如何进入训练和推理系统。

## 建议主题

- Transformer、Attention、MLP、Embedding、LayerNorm、MoE
- Token、Embedding、Logits、Loss、Backpropagation、Optimizer
- 训练循环、推理循环、Autoregressive Generation
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
