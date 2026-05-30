---
title: 基准实验报告模板
domain: template
status: reviewed
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 基准实验报告模板

## 实验目标

说明本次 Benchmark 要回答的问题，例如降低 TTFT、提升 decode throughput、验证分布式扩展效率、比较 Kernel 实现或估算集群容量。

## Workload

- 模型或算子：
- batch size：
- sequence length / input-output tokens：
- precision：
- 并发或节点规模：
- 数据路径：

## 实验环境

| 项目 | 配置 |
| --- | --- |
| 硬件 | GPU/NPU/CPU、显存、网络、存储 |
| 软件 | OS、driver、CUDA/ROCm、framework、runtime、compiler |
| 配置 | 环境变量、启动参数、调度参数 |
| 版本 | Git commit、镜像 tag、依赖版本 |

## 指标

- latency：TTFT、TPOT、p50/p95/p99
- throughput：requests/s、tokens/s、samples/s、step/s
- resource：GPU utilization、SM occupancy、HBM bandwidth、显存占用
- scaling：MFU、scaling efficiency、communication ratio
- reliability：失败率、重试率、timeout、OOM
- energy：power、energy per token、frequency throttling

## 结果

| 实验 | 配置 | 指标 | 结果 | 备注 |
| --- | --- | --- | --- | --- |
| baseline |  |  |  |  |
| variant-a |  |  |  |  |

## 分析

- 主要瓶颈和 profiler 证据。
- 与预期或论文结果的差异。
- 对不同 shape、并发、硬件或版本的敏感性。
- 不确定性、误差来源和统计置信度。

## 结论

- 本次实验支持的结论。
- 不支持或尚不能判断的结论。
- 后续实验和需要沉淀到知识库的条目。
