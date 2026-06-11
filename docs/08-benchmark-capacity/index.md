---
title: 性能分析、Benchmark 与容量建模
domain: benchmark-capacity
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 性能分析、Benchmark 与容量建模

本目录是整个知识库的度量核心。任何“更快、更省、更稳”的结论，都必须回到可复现的指标、实验设计、性能剖析和容量模型。

## 建议主题

- latency、throughput、TTFT、TPOT、p95/p99、tokens/s
- step time、MFU、scaling efficiency、communication ratio
- GPU utilization、SM occupancy、HBM bandwidth、memory footprint
- power、energy per token、thermal、frequency throttling
- profiler：Nsight、PyTorch Profiler、NVIDIA DCGM、perf、eBPF
- Roofline model、queueing model、capacity planning
- Benchmark workload design、warmup、sample size、confidence interval
- A/B comparison、ablation、regression detection

## 关键问题

- 指标定义是否清晰，是否能被别人复现。
- Benchmark workload 是否代表真实请求和训练任务。
- 结果是否区分平均值、尾部、波动和异常值。
- 性能瓶颈是否有 profiler 证据，而不是只看现象。
- 容量模型能否解释现有结果并预测扩容后的表现。

## 专题入口

- [性能分析与 Benchmark 方法论：指标、实验设计与瓶颈定位](performance-analysis-benchmark-methodology.md)：解释 benchmark、profiling、monitoring 的区别，如何从问题出发定义指标、设计 workload、控制变量、处理 warmup、做统计、A/B、ablation、profiler 证据和容量建模输入。
