---
title: 推理过程与原理
domain: inference-primer
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 推理过程与原理

推理是模型参数固定后，根据输入生成输出的过程。AI Infra 关注的是如何让推理低延迟、高吞吐、低成本、稳定且可观测。

## 自回归生成流程

大语言模型通常按 token 逐个生成：

```text
prompt
  -> tokenize
  -> prefill
  -> decode token 1
  -> decode token 2
  -> ...
  -> detokenize
```

每生成一个新 token，模型都会根据已有上下文预测下一个 token 的概率分布，然后通过 greedy、top-k、top-p、temperature 等采样策略选出 token。

## Prefill

Prefill 处理完整 prompt，计算所有输入 token 的表示，并建立 KV Cache。

系统特征：

- 适合较大的矩阵计算。
- prompt 越长，prefill 越重。
- TTFT 很大程度受 prefill、排队和调度影响。

## Decode

Decode 每次生成一个 token，并复用历史 KV Cache。

系统特征：

- 每一步计算规模小，但需要频繁访问 KV Cache。
- 容易受显存带宽、调度和 batch 形态影响。
- TPOT 主要反映 decode 阶段效率。

## KV Cache

KV Cache 保存历史 token 的 key/value，避免每次 decode 重新计算全部历史上下文。

系统影响：

- 长上下文和高并发会让 KV Cache 占用大量显存。
- PagedAttention、prefix cache、cache eviction 都围绕 KV Cache 管理展开。
- KV Cache layout 会影响内存访问效率和 Kernel 性能。

## Batching

推理服务会把多个请求合并运行，提高 GPU 利用率。

- Static batching 简单，但容易等待。
- Dynamic batching 根据请求动态合并。
- Continuous batching 允许请求在 decode 过程中加入或退出。

## 关键指标

| 指标 | 含义 |
| --- | --- |
| TTFT | 从请求进入到第一个 token 返回的时间 |
| TPOT | 每生成一个 token 的平均时间 |
| Throughput | 单位时间生成的 token 或处理的请求 |
| p95/p99 latency | 尾延迟，反映长尾体验 |
| GPU memory | 权重和 KV Cache 等显存占用 |

## 关键问题

- 当前 workload 是 prefill-heavy、decode-heavy 还是混合负载。
- 主要瓶颈是计算、显存容量、显存带宽、排队、调度还是网络。
- batch size 和并发提升后，吞吐增加是否以尾延迟恶化为代价。
- KV Cache 管理策略是否适合目标上下文长度和请求分布。
