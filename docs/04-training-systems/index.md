---
title: 训练系统与优化
domain: training-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-11
---

# 训练系统与优化

本目录关注训练如何在多 GPU、多节点上高效运行。这里不以提升模型任务指标为重点，而以 step time、扩展效率、显存占用、通信开销、故障恢复和复现实验为重点。

## 本章内容安排

训练系统的学习顺序可以按“一次训练 step 如何被放大到多 GPU、多节点”展开：

1. 先理解训练任务生命周期，知道一个 step 由哪些阶段组成。
2. 再学习数据输入、batch、activation、gradient、optimizer state 这些基础成本。
3. 然后进入 data parallel、FSDP / ZeRO、tensor parallel、pipeline parallel、expert parallel 等并行策略。
4. 接着学习通信重叠、FLUX、混合精度、activation checkpointing、checkpoint/restart 和容错。
5. 再理解 optimizer state、Muon 优化器、scheduler 和 optimizer step 的系统成本。
6. 最后用 step time breakdown、MFU、scaling efficiency 和 profiler 证据评估训练效率。

| 顺序 | 主题 | 本章中的作用 |
| --- | --- | --- |
| 1 | [训练任务生命周期](training-lifecycle.md) | 建立一个训练 step 的端到端系统视角。 |
| 2 | [数据输入与 Data Pipeline](data-pipeline.md) | 解释数据读取、tokenization、packing 和 H2D copy 如何影响 GPU 利用率。 |
| 3 | [Batch、Micro-batch 与 Gradient Accumulation](batch-gradient-accumulation.md) | 理解 global batch、显存、吞吐和优化稳定性的关系。 |
| 4 | [显存组成与优化总览](memory-composition-optimization.md) | 拆解 parameters、gradients、optimizer states、activations 和 temporary buffers。 |
| 5 | [Data Parallel 与梯度同步](data-parallel-gradient-sync.md) | 理解数据并行、AllReduce、gradient bucketing 和同步开销。 |
| 6 | [ZeRO 与 FSDP](zero-fsdp.md) | 学习参数、梯度和 optimizer state sharding 如何降低重复显存。 |
| 7 | Tensor Parallel | 学习把单层矩阵和 attention 计算切到多 GPU 的方法。 |
| 8 | Pipeline Parallel | 学习层间切分、micro-batch 流水和 pipeline bubble。 |
| 9 | Expert Parallel 与 MoE 训练 | 处理专家路由、token dispatch、负载均衡和跨卡通信。 |
| 10 | Activation Checkpointing | 用重计算换显存，降低长上下文和大 batch 的 activation 压力。 |
| 11 | 混合精度训练 | 理解 FP16、BF16、FP8、loss scaling 和数值稳定性。 |
| 12 | 通信与计算重叠 | 分析 backward、AllReduce/ReduceScatter、bucket 和 overlap 失败原因。 |
| 13 | FLUX 通信重叠与 Kernel Fusion | 以 FLUX 为案例，理解如何把通信和计算细粒度融合来隐藏分布式通信。 |
| 14 | Optimizer 与 Scheduler 系统成本 | 研究 Adam/AdamW、fused optimizer、学习率调度和 optimizer state 成本。 |
| 15 | Muon 优化器 | 理解矩阵动量正交化优化器的基本思想、适用参数和系统实现成本。 |
| 16 | Checkpoint、Resume 与容错 | 设计长期训练的恢复、存储、sharded checkpoint 和 elastic training。 |
| 17 | 训练性能指标与扩展效率 | 用 step time、tokens/s、MFU、scaling efficiency 和 network utilization 评价训练系统。 |
| 18 | 训练性能剖析与 Benchmark | 用 trace、profiler、通信 timeline 和 ablation 定位训练瓶颈。 |
| 19 | DeepSpeed、Megatron-LM 与 PyTorch FSDP | 作为主流训练系统和框架案例。 |

## 训练任务生命周期

训练任务生命周期说明一个 step 从数据读取、forward、loss、backward、gradient sync、optimizer step 到 checkpoint 的完整链路。它是学习训练系统优化的入口。

详见：[训练任务生命周期](training-lifecycle.md)

## 数据输入与 Data Pipeline

数据输入决定 GPU 是否能持续有活干。训练数据可能经历读取、解压、tokenization、增强、packing、batch 拼接和 host-to-device copy。数据慢会让 GPU 等待。

详见：[数据输入与 Data Pipeline](data-pipeline.md)

## Batch、Micro-batch 与 Gradient Accumulation

训练里的 batch 概念包括 micro-batch、per-device batch、global batch 和 gradient accumulation。它们共同影响显存、吞吐、通信和训练稳定性。

详见：[Batch、Micro-batch 与 Gradient Accumulation](batch-gradient-accumulation.md)

## 显存组成与优化总览

训练显存不只有模型权重，还包括 gradients、optimizer states、activations、temporary buffers 和 allocator fragmentation。不同优化方法减少的显存项不同。

详见：[显存组成与优化总览](memory-composition-optimization.md)

## Data Parallel 与梯度同步

Data Parallel 让多张 GPU 处理不同数据，再同步梯度保持参数一致。它简单有效，但会引入 AllReduce、bucket 和同步等待。

详见：[Data Parallel 与梯度同步](data-parallel-gradient-sync.md)

## ZeRO 与 FSDP

ZeRO 和 FSDP 通过切分参数、梯度和 optimizer state，降低数据并行副本之间的重复显存。它们是大模型训练的重要基础。

详见：[ZeRO 与 FSDP](zero-fsdp.md)

## Tensor Parallel

Tensor Parallel 把单层里的大矩阵计算拆到多张 GPU，常用于单卡放不下或单卡算力不足的大模型训练。

本节后续重点回答：

- column parallel 和 row parallel 的基本思想。
- Tensor Parallel 为什么引入频繁通信。
- 如何避免跨节点 tensor parallel 带来的通信瓶颈。

## Pipeline Parallel

Pipeline Parallel 把模型不同层放到不同 GPU，并用 micro-batch 形成流水线。它能降低单卡权重压力，但会引入 pipeline bubble。

本节后续重点回答：

- pipeline stage 如何划分。
- micro-batch 数量如何影响 bubble。
- 1F1B 等调度如何改善流水线利用率。

## Expert Parallel 与 MoE 训练

MoE 训练中，每个 token 会被路由到部分专家。Expert Parallel 需要处理 token dispatch/combine、专家负载均衡和跨卡通信。

本节后续重点回答：

- MoE 训练为什么比 dense model 更依赖通信和负载均衡。
- expert placement 和 routing 如何影响 step time。
- capacity factor、token dropping 和 load balance loss 如何影响系统行为。

## Activation Checkpointing

Activation Checkpointing 不保存所有中间 activation，而是在 backward 时重算部分 forward。它用额外计算换显存。

本节后续重点回答：

- activation 为什么会占大量显存。
- checkpoint 粒度如何影响重算开销。
- 长上下文训练为什么常需要 activation checkpointing。

## 混合精度训练

混合精度训练用 FP16、BF16、FP8 等低精度降低显存和计算成本，同时要处理 loss scaling、overflow、NaN 和收敛稳定性。

本节后续重点回答：

- FP16、BF16、FP8 在训练中的差异。
- loss scaling 为什么必要。
- 如何同时评估性能提升和数值稳定性。

## 通信与计算重叠

训练通信不一定必须完全阻塞计算。通过 bucket、异步通信和 backward overlap，可以把部分梯度同步隐藏在计算之后。

本节后续重点回答：

- backward 与 AllReduce 如何重叠。
- overlap 失败常见原因是什么。
- 如何用 timeline 判断通信是否真的被隐藏。

## FLUX 通信重叠与 Kernel Fusion

FLUX 是一种把通信和计算切成更细粒度，再融合到更大 GPU kernel 中的通信重叠思路。它的目标不是减少通信量本身，而是让一部分依赖通信的计算能在通信尚未整体结束前就开始，从而把等待时间隐藏起来。

从训练系统角度看，FLUX 适合放在通信重叠之后学习：先理解 AllReduce、ReduceScatter、AllGather、Tensor Parallel 通信为什么会阻塞，再看更激进的 kernel-level overlap 如何改变 timeline。

本节后续重点回答：

- 为什么传统 stream-level overlap 不一定能隐藏依赖通信。
- FLUX 如何用更细粒度分解和 kernel fusion 增加 overlap 空间。
- 它更适合哪些算子、并行策略和硬件互连条件。
- 如何用 profiler 判断 FLUX 类优化是否真的减少了 exposed communication time。

参考资料：

- [FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion](https://arxiv.org/abs/2406.06858)

## Optimizer 与 Scheduler 系统成本

Optimizer step 会更新参数并维护 optimizer state。Adam/AdamW、fused optimizer、学习率 scheduler 和 master weight 都会影响显存和 step time。

本节后续重点回答：

- Adam optimizer state 为什么显存很高。
- fused optimizer 如何降低 kernel launch 和访存开销。
- scheduler 和 optimizer step 如何影响长期训练稳定性。

## Muon 优化器

Muon 是一种面向矩阵参数的优化器思路。直觉上，它不是直接把普通动量矩阵用于更新，而是先对动量矩阵做近似正交化，实践中常用 Newton-Schulz 迭代近似这个过程，再用得到的方向更新权重。

从系统角度看，Muon 值得单独列出，不是因为它一定替代 AdamW，而是因为它把 optimizer 从“逐元素状态更新”推进到“矩阵级更新”。这会带来额外矩阵乘、参数分组、fused 实现、分布式切分和数值稳定性问题。常见实践也不会把所有参数都交给 Muon，例如 embedding、bias、normalization 参数和输出头通常需要单独处理。

本节后续重点回答：

- Muon 和 AdamW、SGD Momentum 的核心差异是什么。
- Newton-Schulz 正交化在 optimizer step 中增加了哪些计算和显存压力。
- 哪些参数适合用 Muon，哪些参数通常仍使用 AdamW 或其他优化器。
- 分布式训练中 Muon 如何影响 sharding、fused optimizer 和 checkpoint。
- 评估 Muon 时应同时看收敛速度、wall-clock time、额外 kernel 成本和稳定性。

参考资料：

- [Convergence of Muon with Newton-Schulz](https://arxiv.org/abs/2601.19156)
- [The Newton-Muon Optimizer](https://arxiv.org/abs/2604.01472)

## Checkpoint、Resume 与容错

长期训练必须考虑 checkpoint、resume、故障恢复和存储压力。checkpoint 不完整或太慢，都会影响训练可靠性。

本节后续重点回答：

- 完整 checkpoint 应包含哪些训练状态。
- sharded checkpoint 如何服务 FSDP/ZeRO 和多节点训练。
- elastic training 和故障恢复如何影响系统设计。

## 训练性能指标与扩展效率

训练系统需要用 step time、samples/s、tokens/s、MFU、GPU memory、communication time 和 scaling efficiency 评价。

本节后续重点回答：

- MFU 和 GPU utilization 有什么区别。
- strong scaling 和 weak scaling 如何评估。
- 扩展效率下降时如何定位原因。

## 训练性能剖析与 Benchmark

训练 Benchmark 不只是跑通脚本，而是固定模型、数据、batch、精度、并行策略和硬件，用 profiler 和 trace 解释 step time。

本节后续重点回答：

- 如何做 step time breakdown。
- 如何用 profiler 区分计算、通信、数据和 checkpoint 瓶颈。
- 如何把训练实验沉淀为容量模型。

## DeepSpeed、Megatron-LM 与 PyTorch FSDP

DeepSpeed、Megatron-LM 和 PyTorch FSDP 是理解大模型训练系统的重要案例。它们分别覆盖 ZeRO、模型并行、分布式 runtime 和主流框架集成。

本节后续重点回答：

- 这些框架分别解决哪些训练系统问题。
- 它们的并行策略、显存策略和 checkpoint 策略有什么不同。
- 如何基于 workload 和硬件做训练框架选型。
