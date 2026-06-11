---
title: 集群、网络、存储与调度
domain: cluster-infra
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 集群、网络、存储与调度

本目录关注 AI Infra 的集群底座：GPU 资源如何被调度、网络和存储如何支撑训练与推理、实验环境如何保持可复现、不同任务如何隔离并共享资源。

## 建议主题

- Slurm、Kubernetes、Ray、Volcano、Kueue
- GPU topology、NUMA、MIG、MPS、资源隔离
- InfiniBand、RoCE、RDMA、NCCL topology、拥塞控制
- 本地 NVMe、并行文件系统、对象存储、数据缓存
- container image、driver、CUDA、library version、environment lock
- queueing、priority、quota、preemption、gang scheduling
- online serving cluster、batch training cluster、mixed workload cluster
- utilization、fragmentation、fairness、SLA、成本和能效指标

## 关键问题

- 调度策略是否让 GPU 空闲或碎片化。
- 网络拓扑和作业拓扑是否匹配。
- 存储吞吐是否让训练或检索任务等待。
- 环境版本是否可复现，失败时是否能快速定位。
- 多租户隔离是否影响性能、稳定性和安全边界。

## 专题入口

- [AI 集群架构总览：节点、网络、存储与调度](ai-cluster-architecture-overview.md)：建立 AI 集群控制面、计算节点、网络平面、存储层次、调度系统、多租户、环境治理和可观测性的整体框架。
