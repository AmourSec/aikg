---
title: AI 系统论文与架构
domain: ai-system-papers
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# AI 系统论文与架构

本页用于记录 AI 系统论文和开源系统的架构分析。每篇笔记都应把论文贡献映射到 workload、runtime、kernel、accelerator、cluster 和 measurement 中的具体层级。

## 建议分析维度

- 论文背景：解决什么系统瓶颈，假设的 workload 是什么。
- 架构位置：推理、训练、Kernel、编译器、硬件、集群还是 Benchmark。
- 核心机制：调度、缓存、并行、融合、量化、通信、存储或容错。
- 实验设置：模型、shape、硬件、软件版本、基线、指标。
- 复现难点：代码、数据、硬件、参数、未公开细节。
- 适用边界：在哪些 workload 或硬件上可能失效。

## 记录格式

| 字段 | 说明 |
| --- | --- |
| Paper | 标题、作者、会议、年份、链接 |
| System Layer | workload / runtime / kernel / accelerator / cluster / measurement |
| Bottleneck | compute / memory / communication / scheduling / reliability |
| Method | 核心方法和实现路径 |
| Evidence | 论文实验、复现实验和 profiler 证据 |
| Limits | 前提、反例和不可迁移之处 |
