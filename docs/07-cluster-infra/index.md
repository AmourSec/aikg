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

- [AI 集群架构总览：节点、网络、存储与调度](ai-cluster-architecture-overview.md)：建立 AI 集群 workload、作业生命周期、resource flavor、容量池、故障域、网络存储、准入/放置/编排、拓扑感知、环境治理、manifest、可观测性和容量规划的整体框架。
- [调度系统与资源队列：Slurm、Kubernetes、Ray、Volcano 与 Kueue](scheduling-queues-resource-management.md)：解释 AI workload 从 job spec、准入、排队、公平共享、resource flavor、capacity pool、拓扑匹配、gang allocation 到抢占恢复、pending reason、策略迭代和可解释性的调度生命周期。
- [GPU 拓扑、NUMA、MIG/MPS 与资源隔离](gpu-topology-numa-mig-mps-isolation.md)：解释 GPU-to-GPU、GPU-to-NIC、CPU NUMA、本地 NVMe、topology manifest、MIG 生命周期、MPS、time slicing、Kubernetes 拓扑组件、Slurm GRES、rank mapping、共享 GPU 治理和故障归因。
- [RDMA 网络与 NCCL 拓扑：InfiniBand、RoCE 与拥塞控制](rdma-network-nccl-topology-congestion.md)：解释 RDMA、GPU Direct RDMA 路径验收、InfiniBand、RoCE、NCCL/RCCL collective、multi-rail、network topology manifest、PFC/ECN/QoS、拥塞域、网络调度契约、故障归因和 benchmark manifest。
- [存储、数据缓存与 Checkpoint：NVMe、并行文件系统与对象存储](storage-data-cache-checkpoint.md)：解释对象存储、并行文件系统、本地 NVMe、数据对象生命周期、存储策略契约、数据集 shard、DataLoader、缓存治理、checkpoint 状态机、模型权重分发防雪崩、容器镜像、GPUDirect Storage、Kubernetes 存储抽象、存储调度契约、故障归因和 benchmark manifest。
- [环境可复现：镜像、驱动、CUDA 与依赖锁定](environment-reproducibility-containers.md)：解释 AI 任务环境的 node profile、image family、run manifest、Driver/CUDA 支持矩阵、镜像供应链与 SBOM、artifact manifest、随机性控制、benchmark 可复现、升级回滚和环境漂移归因。
- [混合集群与多租户隔离：训练、推理、Notebook 与批处理共存](mixed-workload-multitenancy-isolation.md)：解释训练、在线推理、Notebook、数据预处理、benchmark 和系统任务共存时的队列、配额、优先级、抢占、节点池、GPU 共享、存储网络隔离和成本归因。
- [资源利用率、碎片与容量治理：从 GPU 分配到有效吞吐](resource-utilization-fragmentation-capacity.md)：解释 GPU 分配率、活跃率、有效吞吐、排队、pending reason、碎片、公平性、SLA、成本、能效、dashboard、告警和容量规划。
- [节点生命周期与集群运维：交付、验收、入池、维护与下线](node-lifecycle-health-maintenance.md)：解释 AI 计算节点从资产登记、物理验收、软件基线、burn-in、入池、健康检查、cordon/drain、升级、配置漂移、缓存清理到退役下线的运维流程。
