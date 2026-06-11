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
- [调度系统与资源队列：Slurm、Kubernetes、Ray、Volcano 与 Kueue](scheduling-queues-resource-management.md)：解释 AI workload 从提交、准入、排队、公平共享、资源匹配、拓扑匹配、gang allocation 到运行、抢占、重试和可解释性的调度生命周期。
- [GPU 拓扑、NUMA、MIG/MPS 与资源隔离](gpu-topology-numa-mig-mps-isolation.md)：解释 GPU-to-GPU、GPU-to-NIC、CPU NUMA、本地 NVMe、MIG、MPS、time slicing、rank mapping 和多租户 GPU 隔离策略。
- [RDMA 网络与 NCCL 拓扑：InfiniBand、RoCE 与拥塞控制](rdma-network-nccl-topology-congestion.md)：解释 RDMA、GPU Direct RDMA、InfiniBand、RoCE、NCCL/RCCL collective、multi-rail、网络拓扑、PFC/ECN/QoS、拥塞排查和网络 benchmark。
- [存储、数据缓存与 Checkpoint：NVMe、并行文件系统与对象存储](storage-data-cache-checkpoint.md)：解释对象存储、并行文件系统、本地 NVMe、数据集 shard、DataLoader、模型权重分发、容器镜像、GPUDirect Storage、checkpoint 原子性、异步保存和恢复。
- [环境可复现：镜像、驱动、CUDA 与依赖锁定](environment-reproducibility-containers.md)：解释 AI 任务环境的硬件、host driver、CUDA、容器镜像、Python/Conda lock、数据模型 artifact、run manifest、随机性控制和升级验证。
- [混合集群与多租户隔离：训练、推理、Notebook 与批处理共存](mixed-workload-multitenancy-isolation.md)：解释训练、在线推理、Notebook、数据预处理、benchmark 和系统任务共存时的队列、配额、优先级、抢占、节点池、GPU 共享、存储网络隔离和成本归因。
- [资源利用率、碎片与容量治理：从 GPU 分配到有效吞吐](resource-utilization-fragmentation-capacity.md)：解释 GPU 分配率、活跃率、有效吞吐、排队、pending reason、碎片、公平性、SLA、成本、能效、dashboard、告警和容量规划。
