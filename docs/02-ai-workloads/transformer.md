---
title: Transformer 流程与原理
domain: transformer
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# Transformer 流程与原理

Transformer 是当前大语言模型和很多多模态模型的核心结构。这里关注它的基本数据流和系统含义，帮助读者理解后续的推理、训练、Kernel 和硬件优化。

## 基本流程

```text
Token IDs
  -> Token Embedding + Position Information
  -> Transformer Block x N
       -> Attention
       -> MLP / FFN
       -> Residual + LayerNorm
  -> Final Norm
  -> LM Head
  -> Logits
```

每一层 Transformer Block 通常包含两个主要计算部分：Attention 和 MLP。Attention 负责让当前位置读取上下文信息，MLP 负责对每个位置的表示做非线性变换。

## Attention 在做什么

Attention 会从输入向量生成三组向量：Q、K、V。

```text
Q = X Wq
K = X Wk
V = X Wv
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

直观理解：

- Q 表示当前位置想查什么。
- K 表示每个历史位置能被怎样匹配。
- V 表示真正被读取的信息。
- `QK^T` 会比较所有 token 之间的相关性。

## 为什么 Attention 影响性能

- Prefill 阶段会处理整段 prompt，Attention 计算和 sequence length 强相关。
- Decode 阶段每次生成一个 token，但需要读取历史 KV Cache。
- 长上下文会让 KV Cache 显存和带宽压力变大。
- Attention 的矩阵乘、softmax、mask、layout 对 Kernel 优化很敏感。

## MLP 在做什么

MLP 通常是两到三个线性层加激活函数。它的计算量很大，经常以 GEMM 为主。

```text
X -> Linear Up -> Activation -> Linear Down
```

系统视角下，MLP 通常更偏 compute-bound，适合利用 Tensor Core；Attention 和归一化类算子更容易受内存访问、layout 和 fusion 影响。

## 关键问题

- 当前瓶颈来自 Attention、MLP、Embedding、LayerNorm 还是输出层。
- 问题发生在 prefill 还是 decode。
- sequence length、hidden size、head 数、batch size 如何改变计算量和显存。
- Kernel 或编译器是否针对当前 shape 做了合适优化。
