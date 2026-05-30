---
title: 训练系统与分布式计算
domain: training-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 训练系统与分布式计算

本目录关注训练如何在多 GPU、多节点上高效运行。这里不以提升模型任务指标为重点，而以 step time、扩展效率、显存占用、通信开销、故障恢复和复现实验为重点。

## 建议主题

- data parallel、tensor parallel、pipeline parallel、expert parallel
- ZeRO、FSDP、DeepSpeed、Megatron-LM、NCCL
- activation checkpointing、gradient accumulation、optimizer state sharding
- AllReduce、AllGather、ReduceScatter、通信重叠
- pipeline bubble、load balance、straggler、MoE dispatch
- checkpoint、resume、fault tolerance、elastic training
- mixed precision、loss scaling、determinism、random seed
- step time breakdown、MFU、scaling efficiency、network utilization

## 关键问题

- 训练瓶颈来自计算、显存容量、内存带宽、跨卡通信还是数据输入。
- 并行策略如何影响显存占用、通信量和实现复杂度。
- 通信是否能与计算重叠，重叠失败时原因是什么。
- Checkpoint 策略如何影响故障恢复时间和存储压力。
- 扩展效率下降时，是网络、负载均衡、同步点还是输入 pipeline 导致。
