---
title: 训练过程与原理
domain: training-primer
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 训练过程与原理

训练的目标是让模型参数逐步变得更适合目标数据。AI Infra 关注的是训练如何消耗计算、显存、网络和存储，以及如何把训练过程稳定、可复现地跑起来。

## 单步训练流程

```text
读取 batch
  -> tokenizer / data loader
  -> forward
  -> loss
  -> backward
  -> gradient sync
  -> optimizer step
  -> checkpoint / logging
```

## Forward

Forward 是模型从输入 token 计算 logits 和 loss 的过程。训练时不仅要算出结果，还要保存大量中间 activation，供 backward 计算梯度。

系统影响：

- activation 会占用大量显存。
- sequence length 和 batch size 越大，显存和计算压力越大。
- activation checkpointing 可以省显存，但会增加重算开销。

## Backward

Backward 根据 loss 计算每个参数的 gradient。它通常比 forward 更耗时，并且会产生梯度张量。

系统影响：

- gradient 需要显存。
- 多 GPU 训练需要同步 gradient。
- 通信库、网络拓扑和并行策略会影响 step time。

## Optimizer

Optimizer 用 gradient 更新参数。Adam 这类优化器会保存额外状态，例如一阶矩和二阶矩。

系统影响：

- optimizer state 可能比参数本身还占显存。
- ZeRO、FSDP 会切分参数、梯度和 optimizer state。
- checkpoint 需要保存参数和优化器状态，带来存储压力。

## 并行训练为什么复杂

单卡放不下或太慢时，需要并行：

- Data Parallel：每张卡处理不同 batch，梯度同步。
- Tensor Parallel：把单层矩阵计算切到多张卡。
- Pipeline Parallel：把不同层放到不同卡。
- Expert Parallel：MoE 中不同 expert 分布到不同卡。

## 关键问题

- step time 主要花在 forward、backward、通信还是数据读取。
- 显存主要被参数、activation、gradient 还是 optimizer state 占用。
- 并行策略是否带来过多通信或 pipeline bubble。
- checkpoint、resume、随机性和版本是否支持复现。
