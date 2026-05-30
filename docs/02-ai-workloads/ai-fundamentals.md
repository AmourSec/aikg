---
title: AI 基础概念
domain: ai-fundamentals
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# AI 基础概念

本页面向刚开始接触 AI Infra 的读者，解释最小必要的 AI 基础。目标不是完整学习机器学习理论，而是能看懂训练、推理、Transformer、Benchmark 和系统论文里的基本词汇。

## 最小概念表

| 概念 | 含义 | 系统视角 |
| --- | --- | --- |
| Tensor | 多维数组，是模型计算的基本数据结构 | shape 决定算子规模和内存访问 |
| Token | 文本被 tokenizer 切分后的离散单位 | token 数决定序列长度和推理成本 |
| Embedding | 把 token id 映射成向量 | 主要是查表和内存访问 |
| Parameter | 模型中被训练出来的权重 | 参数量决定显存、加载时间和通信量 |
| Activation | forward 中间结果 | 训练时需要保存，显存压力很大 |
| Logits | 模型输出的未归一化分数 | 推理时用于选下一个 token |
| Loss | 训练目标函数 | backward 从 loss 开始传播梯度 |
| Gradient | loss 对参数的导数 | 训练时需要额外显存和通信 |
| Optimizer | 用梯度更新参数的算法 | Adam 等优化器会保存额外状态 |

## 一个模型在做什么

语言模型的基本任务可以粗略理解为：给定前面的 token，预测下一个 token 的概率分布。

```text
输入文本 -> Token IDs -> Embedding -> 多层网络 -> Logits -> 下一个 token 概率
```

训练时，模型看到大量文本，比较预测和真实下一个 token 的差距，用 loss 更新参数。推理时，模型没有真实答案，只能反复预测下一个 token，再把生成的新 token 接回输入。

## 为什么 Infra 需要懂这些

- token 数决定 sequence length，直接影响 Attention 计算量和 KV Cache 显存。
- 参数量决定权重加载、显存容量、分布式切分和通信量。
- activation 和 gradient 主要影响训练显存。
- logits、sampling 和 decode loop 主要影响推理延迟。
- shape、precision、batch size 决定 Kernel、编译器和硬件能否跑满。

## 关键问题

- 当前问题发生在训练还是推理。
- 输入是固定长度、长上下文、动态 batch 还是多轮对话。
- 主要资源瓶颈是参数、激活、KV Cache、梯度还是优化器状态。
- 模型输出质量问题和系统性能问题是否被混在一起讨论。
