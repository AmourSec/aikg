---
title: AI 计算工作负载基础
domain: ai-workloads
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-04
---

# AI 计算工作负载基础

本目录是 AI 入门科普层，只解决一个问题：**让完全不熟悉 AI 的读者先知道模型、Transformer、训练、推理和多模态大概是怎么回事。**

这里不深入讲公式、调参、并行训练、推理优化、硬件架构或性能分析。读完后，读者应该能用自己的话说清楚：

- AI 模型为什么能从数据里学到规律。
- Transformer 为什么能理解一段上下文。
- 训练过程为什么能让模型慢慢变好。
- 推理过程为什么能根据提示词生成回答。
- 多模态模型为什么能把图片、语音、视频和文字放在一起处理。

## 入门顺序

| 顺序 | 页面 | 读完应该懂什么 |
| --- | --- | --- |
| 1 | [AI 基础概念](ai-fundamentals.md) | 模型、数据、参数、token、embedding、tensor、logits、loss、梯度、batch 和上下文是什么。 |
| 2 | [Transformer 流程与原理](transformer.md) | 一段文字如何经过 token、embedding、位置、self-attention、Q/K/V、causal mask、MLP 和 logits，最后变成下一个 token 的概率。 |
| 3 | [训练过程与原理](training-primer.md) | 模型如何先猜答案、计算错误、根据错误调整参数，并不断重复。 |
| 4 | [推理过程与原理](inference-primer.md) | 模型参数固定后，如何读取 prompt，并一个 token 一个 token 生成输出。 |
| 5 | [多模态原理](multimodal-primer.md) | 多模态理解如何读懂图片、音频、视频，多模态生成如何生成新内容。 |

## 本目录的讲解边界

本目录只讲到“理论上为什么可行”和“流程上每一步做什么”。暂时不展开：

- 训练如何并行到很多 GPU。
- 推理服务如何提升吞吐和降低延迟。
- Attention、KV Cache、batch、精度、显存的性能细节。
- Triton、TorchInductor、FlashAttention 等工程优化。
- 模型结构改造、微调方法和任务指标提升。

这些内容会放到后续系统、Kernel、硬件、集群和 Benchmark 章节里。这里先把地基打清楚。

## 最小主线

可以先把大语言模型理解成下面这条链路：

```text
很多文本样本
  -> 训练：让模型学会根据前文预测后文
  -> 得到一组固定参数
  -> 推理：给模型一段 prompt
  -> 模型预测下一个 token
  -> 把新 token 接回上下文，继续预测
  -> 得到完整回答
```

如果只记一句话：

> 训练是在大量样本上调参数；推理是在参数固定后按上下文预测下一个 token；Transformer 是当前最常用的“读上下文并生成表示”的模型结构。

## 学完后再往哪里走

学完这 5 页后，再进入后续章节会更顺：

- 想理解在线大模型服务，读 `推理系统与优化`。
- 想理解大规模训练，读 `训练系统与优化`。
- 想理解 Triton、Inductor、算子优化，读 `Kernel、算子与编译优化`。
- 想理解 GPU、NPU、内存和互连，读 `AI 加速器与计算架构`。
