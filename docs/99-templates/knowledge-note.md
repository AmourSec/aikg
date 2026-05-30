---
title: 知识点模板
domain: template
status: reviewed
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 知识点模板

## 摘要

用 3 到 5 句话说明这个知识点解决的 AI Infra 问题、所属系统层级和主要结论。

## 元数据

| 字段 | 内容 |
| --- | --- |
| domain | inference / training / kernel / compiler / accelerator / cluster / benchmark / reliability |
| system_layer | workload / runtime / kernel / accelerator / cluster / measurement |
| workload | 模型、shape、batch、sequence length、precision、并发 |
| hardware | GPU/NPU/CPU、内存、网络、存储 |
| software | framework、runtime、driver、compiler、library version |
| source | 论文、代码、实验、故障单或讨论记录 |
| status | draft / reviewed / verified / deprecated |

## 背景

- 问题出现在哪个系统层级。
- 目标是降低延迟、提升吞吐、减少显存、提高能效、改善稳定性还是增强复现性。
- 当前已知约束是什么。

## 核心内容

- 关键概念。
- 机制说明。
- 公式、伪代码或执行链路。
- 与上游 workload 和下游系统的关系。

## 证据

- Benchmark 或 profiler 结果。
- 论文、代码或硬件规格来源。
- 适用范围和不适用场景。

## 相关链接

- 上游知识点：
- 下游知识点：
- 相关论文或代码：
