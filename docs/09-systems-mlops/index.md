---
title: 分布式系统与实验平台
domain: systems-mlops
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 分布式系统与实验平台

本目录整理支撑 AI 研究的分布式训练系统、集群调度、存储网络和实验平台。重点是让实验可运行、可观察、可复现、可扩展。

## 建议主题

- 数据并行、张量并行、流水线并行、专家并行
- AllReduce、AllGather、ReduceScatter、通信重叠
- Slurm、Kubernetes、Ray、MPI、训练任务调度
- Checkpoint、容错、弹性训练、作业恢复
- 共享存储、数据缓存、数据加载瓶颈
- 实验追踪、环境管理、结果归档和复现实验平台

## 关键问题

- 分布式训练扩展效率如何度量
- 训练失败时能否恢复并保留可解释日志
- 实验环境、代码版本、数据版本和硬件拓扑是否可追踪
