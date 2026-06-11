---
title: 训练系统与优化
domain: training-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# 训练系统与优化

本目录关注训练如何在多 GPU、多节点上高效运行。这里不以提升模型任务指标为重点，而以 step time、扩展效率、显存占用、通信开销、故障恢复和复现实验为重点。

## 本章内容安排

训练系统的学习顺序可以按“一次训练 step 如何被放大到多 GPU、多节点”展开：

1. 先理解训练任务生命周期，知道一个 step 由哪些阶段组成。
2. 再学习数据输入、batch、activation、gradient、optimizer state 这些基础成本。
3. 然后理解分布式训练 runtime，再进入 data parallel、FSDP / ZeRO、tensor parallel、pipeline parallel、expert parallel，以及这些并行维度如何组合。
4. 接着学习混合精度、数值稳定性、通信重叠、FLUX、activation checkpointing、checkpoint/restart 和容错。
5. 再理解 optimizer state、参数高效微调、后训练工作负载、Muon 优化器、scheduler 和 optimizer step 的系统成本。
6. 最后用 step time breakdown、MFU、scaling efficiency 和 profiler 证据评估训练效率。

| 顺序 | 主题 | 本章中的作用 |
| --- | --- | --- |
| 1 | [训练任务生命周期](training-lifecycle.md) | 建立一个训练 step 的端到端系统视角。 |
| 2 | [数据输入与 Data Pipeline](data-pipeline.md) | 解释数据读取、tokenization、packing 和 H2D copy 如何影响 GPU 利用率。 |
| 3 | [Batch、Micro-batch 与 Gradient Accumulation](batch-gradient-accumulation.md) | 理解 global batch、显存、吞吐和优化稳定性的关系。 |
| 4 | [显存组成与优化总览](memory-composition-optimization.md) | 拆解 parameters、gradients、optimizer states、activations 和 temporary buffers。 |
| 5 | [分布式训练启动与运行时：torchrun、Rank、Process Group 与 NCCL](distributed-training-runtime.md) | 理解 launcher、rank、world size、local rank、rendezvous、process group、backend 和 NCCL 如何支撑分布式训练。 |
| 6 | [Data Parallel 与梯度同步](data-parallel-gradient-sync.md) | 理解数据并行、AllReduce、gradient bucketing 和同步开销。 |
| 7 | [ZeRO 与 FSDP](zero-fsdp.md) | 学习参数、梯度和 optimizer state sharding 如何降低重复显存。 |
| 8 | [Tensor Parallel](tensor-parallel.md) | 学习把单层矩阵和 attention 计算切到多 GPU 的方法。 |
| 9 | [Sequence Parallel 与 Context Parallel](sequence-context-parallel.md) | 理解长序列训练中如何切分 token 序列或上下文，降低 activation 和 attention 压力。 |
| 10 | [Pipeline Parallel](pipeline-parallel.md) | 学习层间切分、micro-batch 流水和 pipeline bubble。 |
| 11 | [Expert Parallel 与 MoE 训练](expert-parallel-moe-training.md) | 处理专家路由、token dispatch、负载均衡和跨卡通信。 |
| 12 | [并行策略组合：3D/4D/5D Parallelism](hybrid-parallelism-composition.md) | 把 DP/FSDP、TP、PP、EP、SP/CP 放到同一个 rank topology 中理解。 |
| 13 | [Activation Checkpointing](activation-checkpointing.md) | 用重计算换显存，降低长上下文和大 batch 的 activation 压力。 |
| 14 | [混合精度训练](mixed-precision-training.md) | 理解 FP16、BF16、FP8、loss scaling 和数值稳定性。 |
| 15 | [训练稳定性与数值异常：NaN、Inf、Loss Spike 与梯度健康](training-stability-numerical-debugging.md) | 把 NaN/Inf、loss spike、grad norm、loss scale、bad batch、checkpoint rollback 和 stability guardrail 纳入训练系统。 |
| 16 | [通信与计算重叠](communication-computation-overlap.md) | 分析 backward、AllReduce/ReduceScatter、bucket 和 overlap 失败原因。 |
| 17 | [FLUX 通信重叠与 Kernel Fusion](flux-kernel-fusion.md) | 以 FLUX 为案例，理解如何把通信和计算细粒度融合来隐藏分布式通信。 |
| 18 | [Optimizer 与 Scheduler 系统成本](optimizer-scheduler-cost.md) | 研究 Adam/AdamW、fused optimizer、学习率调度和 optimizer state 成本。 |
| 19 | [参数高效微调：LoRA、QLoRA 与 Adapter 系统优化](parameter-efficient-finetuning-lora-qlora.md) | 理解只训练少量 adapter 参数时，显存、optimizer state、checkpoint、分布式和推理服务如何变化。 |
| 20 | [后训练工作负载：SFT、DPO、RLHF 与 GRPO 系统视角](post-training-workloads-sft-dpo-rlhf-grpo.md) | 理解后训练如何把监督微调、偏好优化、在线 rollout、reward/verifier 和 policy update 组合成不同系统负载。 |
| 21 | [Muon 优化器](muon-optimizer.md) | 理解矩阵动量正交化优化器的基本思想、适用参数和系统实现成本。 |
| 22 | [Checkpoint、Resume 与容错](checkpoint-resume-fault-tolerance.md) | 设计长期训练的恢复、存储、sharded checkpoint 和 elastic training。 |
| 23 | [训练性能指标与扩展效率](training-performance-metrics-scaling.md) | 用 step time、tokens/s、MFU、scaling efficiency 和 network utilization 评价训练系统。 |
| 24 | [训练性能剖析与 Benchmark](training-benchmark-profiling.md) | 用 trace、profiler、通信 timeline 和 ablation 定位训练瓶颈。 |
| 25 | [DeepSpeed、Megatron-LM 与 PyTorch FSDP](deepspeed-megatron-fsdp.md) | 作为主流训练系统和框架案例。 |

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

## 分布式训练启动与运行时

分布式训练不是一个 Python 进程变大，而是一组进程通过 rank、process group 和 backend 协作。理解 torchrun、rendezvous、local rank、NCCL、环境变量、rank-aware logging 和常见启动失败，是学习 DDP/FSDP/TP/PP/EP 的基础。

详见：[分布式训练启动与运行时：torchrun、Rank、Process Group 与 NCCL](distributed-training-runtime.md)

## Data Parallel 与梯度同步

Data Parallel 让多张 GPU 处理不同数据，再同步梯度保持参数一致。它简单有效，但会引入 AllReduce、bucket 和同步等待。

详见：[Data Parallel 与梯度同步](data-parallel-gradient-sync.md)

## ZeRO 与 FSDP

ZeRO 和 FSDP 通过切分参数、梯度和 optimizer state，降低数据并行副本之间的重复显存。它们是大模型训练的重要基础。

详见：[ZeRO 与 FSDP](zero-fsdp.md)

## Tensor Parallel

Tensor Parallel 把单层里的大矩阵计算拆到多张 GPU，常用于单卡放不下或单卡算力不足的大模型训练。

详见：[Tensor Parallel](tensor-parallel.md)

## Sequence Parallel 与 Context Parallel

长序列训练中，瓶颈不一定来自参数量，也可能来自 sequence length 带来的 activation、attention、mask、position id 和 micro-batch 压力。Sequence Parallel 和 Context Parallel 关注如何切分 token 序列或上下文。

详见：[Sequence Parallel 与 Context Parallel](sequence-context-parallel.md)

## Pipeline Parallel

Pipeline Parallel 把模型不同层放到不同 GPU，并用 micro-batch 形成流水线。它能降低单卡权重压力，但会引入 pipeline bubble。

详见：[Pipeline Parallel](pipeline-parallel.md)

## Expert Parallel 与 MoE 训练

MoE 训练中，每个 token 会被路由到部分专家。Expert Parallel 需要处理 token dispatch/combine、专家负载均衡和跨卡通信。

详见：[Expert Parallel 与 MoE 训练](expert-parallel-moe-training.md)

## 并行策略组合：3D/4D/5D Parallelism

真实大模型训练通常不是只用一种并行策略，而是把 DP/FSDP、TP、PP、EP、SP/CP 等维度组合起来。组合时最关键的是每个维度切分什么、通信发生在哪里，以及 rank mapping 是否匹配硬件拓扑。

详见：[并行策略组合：3D/4D/5D Parallelism](hybrid-parallelism-composition.md)

## Activation Checkpointing

Activation Checkpointing 不保存所有中间 activation，而是在 backward 时重算部分 forward。它用额外计算换显存。

详见：[Activation Checkpointing](activation-checkpointing.md)

## 混合精度训练

混合精度训练用 FP16、BF16、FP8 等低精度降低显存和计算成本，同时要处理 loss scaling、overflow、NaN 和收敛稳定性。

详见：[混合精度训练](mixed-precision-training.md)

## 训练稳定性与数值异常

训练系统优化不能只看速度。NaN、Inf、loss spike、梯度爆炸、loss scale 频繁下降、optimizer state 污染和坏 checkpoint 都会浪费算力。稳定性治理需要把 loss、grad norm、precision state、rank、batch、checkpoint 和恢复策略串起来。

详见：[训练稳定性与数值异常：NaN、Inf、Loss Spike 与梯度健康](training-stability-numerical-debugging.md)

## 通信与计算重叠

训练通信不一定必须完全阻塞计算。通过 bucket、异步通信和 backward overlap，可以把部分梯度同步隐藏在计算之后。

详见：[通信与计算重叠](communication-computation-overlap.md)

## FLUX 通信重叠与 Kernel Fusion

FLUX 是一种把通信和计算切成更细粒度，再融合到更大 GPU kernel 中的通信重叠思路。它的目标不是减少通信量本身，而是让一部分依赖通信的计算能在通信尚未整体结束前就开始，从而把等待时间隐藏起来。

详见：[FLUX 通信重叠与 Kernel Fusion](flux-kernel-fusion.md)

## Optimizer 与 Scheduler 系统成本

Optimizer step 会更新参数并维护 optimizer state。Adam/AdamW、fused optimizer、学习率 scheduler 和 master weight 都会影响显存和 step time。

详见：[Optimizer 与 Scheduler 系统成本](optimizer-scheduler-cost.md)

## 参数高效微调：LoRA、QLoRA 与 Adapter 系统优化

参数高效微调通过冻结基础模型、只训练少量 adapter 参数，降低 gradients、optimizer states 和 checkpoint 成本。LoRA、QLoRA 和 adapter 管理不仅是算法技巧，也会影响训练显存、分布式策略、微调平台、artifact registry 和推理服务 cache key。

详见：[参数高效微调：LoRA、QLoRA 与 Adapter 系统优化](parameter-efficient-finetuning-lora-qlora.md)

## 后训练工作负载：SFT、DPO、RLHF 与 GRPO 系统视角

后训练工作负载覆盖 SFT、reward model、DPO、RLHF/PPO、GRPO 等训练形态。它们不只是“再训练几轮”，而会引入 chosen/rejected 成对数据、reference logprob、在线 rollout、reward/verifier scoring、policy update、rollout 队列和样本版本管理。

详见：[后训练工作负载：SFT、DPO、RLHF 与 GRPO 系统视角](post-training-workloads-sft-dpo-rlhf-grpo.md)

## Muon 优化器

Muon 是一种面向矩阵参数的优化器思路。直觉上，它不是直接把普通动量矩阵用于更新，而是先对动量矩阵做近似正交化，实践中常用 Newton-Schulz 迭代近似这个过程，再用得到的方向更新权重。

从系统角度看，Muon 值得单独列出，不是因为它一定替代 AdamW，而是因为它把 optimizer 从“逐元素状态更新”推进到“矩阵级更新”。这会带来额外矩阵乘、参数分组、fused 实现、分布式切分和数值稳定性问题。常见实践也不会把所有参数都交给 Muon，例如 embedding、bias、normalization 参数和输出头通常需要单独处理。

详见：[Muon 优化器](muon-optimizer.md)

## Checkpoint、Resume 与容错

长期训练必须考虑 checkpoint、resume、故障恢复和存储压力。checkpoint 不完整或太慢，都会影响训练可靠性。

详见：[Checkpoint、Resume 与容错](checkpoint-resume-fault-tolerance.md)

## 训练性能指标与扩展效率

训练系统需要用 step time、samples/s、tokens/s、MFU、GPU memory、communication time 和 scaling efficiency 评价。

详见：[训练性能指标与扩展效率](training-performance-metrics-scaling.md)

## 训练性能剖析与 Benchmark

训练 Benchmark 不只是跑通脚本，而是固定模型、数据、batch、精度、并行策略和硬件，用 profiler 和 trace 解释 step time。

详见：[训练性能剖析与 Benchmark](training-benchmark-profiling.md)

## DeepSpeed、Megatron-LM 与 PyTorch FSDP

DeepSpeed、Megatron-LM 和 PyTorch FSDP 是理解大模型训练系统的重要案例。它们分别覆盖 ZeRO、模型并行、分布式 runtime 和主流框架集成。

详见：[DeepSpeed、Megatron-LM 与 PyTorch FSDP](deepspeed-megatron-fsdp.md)
