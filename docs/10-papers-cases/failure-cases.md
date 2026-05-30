---
title: 故障复盘
domain: failure-cases
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 故障复盘

本页用于沉淀 AI Infra 方向的失败案例、线上故障、复现实验失败和性能回归。复盘不是追责，而是把故障转化为测试用例、监控项、Runbook 和知识库条目。

## 建议主题

- 推理服务延迟抖动、吞吐下降、KV Cache OOM
- NCCL timeout、训练 hang、checkpoint 损坏
- Kernel 性能回归、编译器误优化、driver 或 CUDA 版本问题
- 存储带宽不足、数据读取阻塞、网络拥塞
- GPU 异常、ECC、Xid、温度或功耗限制
- 调度错误、资源碎片、多租户互相影响

## 复盘结构

- 现象：用户可见影响和关键指标变化。
- 时间线：发现、定位、缓解、恢复。
- 根因：workload、runtime、kernel、hardware、cluster 或 process。
- 证据：日志、metrics、profile、实验复现。
- 改进：监控、告警、测试、容量模型、文档和流程。

## 关键问题

- 这类故障是否能被提前发现。
- 是否可以构造成回归测试或 Benchmark case。
- Runbook 是否能帮助新同学独立处理同类问题。
- 故障结论是否能被 AI 检索系统准确引用。
