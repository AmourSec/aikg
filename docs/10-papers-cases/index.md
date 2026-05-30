---
title: 论文复现与系统案例
domain: papers-cases
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 论文复现与系统案例

本目录用于沉淀 AI 系统论文、开源系统、架构案例、复现实验和技术决策。重点不是记录“读过什么”，而是把论文方法转化为可验证的系统知识。

## 建议主题

- 推理系统论文：serving、batching、KV Cache、speculative decoding
- 训练系统论文：parallelism、communication、checkpoint、elastic training
- Kernel 与编译论文：operator fusion、attention kernel、auto-tuning
- 加速器论文：architecture、memory hierarchy、interconnect、precision
- 集群系统论文：scheduling、network、storage、multi-tenancy
- 复现报告：环境、workload、指标、结果、差异和失败原因
- 技术决策：方案选择、benchmark 证据、约束和后续验证

## 关键问题

- 论文解决的是哪个系统瓶颈。
- 方法依赖的 workload、硬件和软件前提是什么。
- 复现结果是否达到论文水平，差异来自哪里。
- 方法能否迁移到不同模型、不同 shape、不同硬件或不同集群规模。
- 案例是否给后续工程或研究留下可复用结论。
