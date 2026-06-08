---
title: 推理系统与优化
domain: inference-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-08
---

# 推理系统与优化

本目录关注模型进入在线服务后的系统问题：如何降低延迟、提高吞吐、控制显存、隔离租户、处理长尾请求，并让服务行为可测、可复现。

## 本章内容安排

推理系统的学习顺序可以按一条完整链路展开：

1. 先理解一个请求从进入服务到输出 token 的生命周期。
2. 再拆开 Prefill 和 Decode，理解两类计算为什么瓶颈不同。
3. 然后学习指标、Batching、KV Cache、调度和显存管理。
4. 接着进入量化、Speculative Decoding、MoE、分离部署和分布式推理。
5. 最后用 vLLM、TensorRT-LLM、SGLang、RAG / Agent 和 Benchmark 把概念落到工程系统上。

| 顺序 | 主题 | 本章中的作用 |
| --- | --- | --- |
| 1 | [推理请求生命周期](request-lifecycle.md) | 建立端到端视角，知道请求经过哪些系统环节。 |
| 2 | [Prefill 与 Decode](prefill-decode.md) | 理解 LLM 推理的两阶段结构和不同瓶颈。 |
| 3 | [指标体系](metrics.md) | 用 TTFT、TPOT、吞吐、尾延迟、显存和成本描述优化目标。 |
| 4 | [Batching](batching.md) | 理解为什么合批能提高吞吐，以及为什么会影响延迟。 |
| 5 | [KV Cache](kv-cache.md) | 解释 Decode 阶段为什么需要缓存历史上下文。 |
| 6 | PagedAttention | 理解块式 KV Cache 管理如何降低显存浪费。 |
| 7 | Prefix Cache | 理解共享 prompt 前缀如何减少重复 Prefill。 |
| 8 | 调度策略 | 研究请求队列、优先级、准入控制、抢占和公平性。 |
| 9 | 量化推理 | 用更低精度减少显存、带宽和计算开销。 |
| 10 | Speculative Decoding | 用草稿模型或多 token 预测减少串行解码等待。 |
| 11 | Prefill/Decode 分离部署 | 将两类阶段放到不同资源池，缓解互相干扰。 |
| 12 | MoE 模型推理优化 | 处理专家路由、负载均衡、通信和显存问题。 |
| 13 | 单机推理服务架构 | 梳理一台机器上的模型加载、执行、队列和 API 服务。 |
| 14 | 多机分布式推理 | 扩展到多 GPU、多节点和跨节点并行。 |
| 15 | 缓存体系 | 统一理解 query、embedding、prefix、KV、tool result 等缓存。 |
| 16 | Benchmark 方法 | 设计可复现实验，避免只看单个吞吐数字。 |
| 17 | vLLM | 作为现代开源推理引擎的主线案例。 |
| 18 | TensorRT-LLM | 作为 NVIDIA 高性能推理栈案例。 |
| 19 | SGLang | 作为结构化生成和高性能 runtime 案例。 |
| 20 | RAG / Agent 推理负载 | 研究复合推理链路如何改变延迟、吞吐和可靠性。 |
| 21 | Benchmark 方法与性能剖析 | 把压测、profiling 和容量分析连接起来。 |

## 推理请求生命周期

推理请求生命周期说明一个在线请求从 API 接入到最终释放资源的完整链路，包括鉴权、tokenization、排队、调度、Prefill、KV Cache、Decode、流式返回和指标记录。

详见：[推理请求生命周期](request-lifecycle.md)

## Prefill 与 Decode

Prefill 与 Decode 说明 LLM 推理为什么分成“先读完输入”和“逐 token 生成输出”两个阶段，以及两者如何分别影响 TTFT、TPOT、显存、调度和在线服务体验。

详见：[Prefill 与 Decode](prefill-decode.md)

## 指标体系

推理优化必须先定义目标。常见指标包括 TTFT、TPOT、end-to-end latency、p50/p95/p99、tokens/s、requests/s、GPU memory、GPU utilization、cost per token 和 goodput。

详见：[指标体系](metrics.md)

## Batching

Batching 的核心是把多个请求合在一起执行，让 GPU 一次处理更多工作。传统 static batching 会被最慢请求拖住，现代推理系统更多使用 dynamic batching、continuous batching 或 iteration-level scheduling。

详见：[Batching](batching.md)

## KV Cache

KV Cache 保存历史 token 的 key/value 表示，让 Decode 阶段不用每生成一个 token 都重新计算全部上下文。它是 LLM 推理显存占用和调度复杂度的核心来源之一。

详见：[KV Cache](kv-cache.md)

## PagedAttention

PagedAttention 把 KV Cache 管理成固定大小的块，类似操作系统里的分页思想。它的价值在于减少显存碎片和重复复制，让系统能容纳更多并发请求。

本节后续重点回答：

- 连续 KV Cache 分配为什么容易浪费显存。
- block table、physical block、copy-on-write 分别解决什么问题。
- PagedAttention 与 continuous batching、prefix cache 如何配合。

## Prefix Cache

Prefix Cache 复用不同请求之间相同的 prompt 前缀，例如 system prompt、工具说明、few-shot 示例或固定 RAG 模板。命中后可以跳过重复 Prefill，降低 TTFT 和 GPU 计算量。

本节后续重点回答：

- 哪些业务场景容易产生可复用前缀。
- prefix cache 命中率如何影响收益。
- prefix cache 与 KV Cache、路由策略、模板规范有什么关系。

## 调度策略

推理调度决定哪些请求先进入 GPU、哪些请求等待、哪些请求被拒绝、哪些请求被迁移或拆分。调度策略会直接影响吞吐、尾延迟、公平性和资源利用率。

本节后续重点回答：

- FCFS、priority、SLO-aware、cache-aware routing 适合什么场景。
- admission control、rate limit、queueing、backpressure 如何保护系统。
- 高并发下如何避免少数长请求拖慢所有请求。

## 量化推理

量化推理用更低精度表示权重、激活或 KV Cache，目标是减少显存占用、内存带宽压力和计算开销。常见方向包括 FP8、INT8、INT4、AWQ、GPTQ 和 weight-only quantization。

本节后续重点回答：

- 权重量化、激活量化和 KV Cache 量化分别影响什么。
- 量化为什么可能提升吞吐，也可能影响质量或稳定性。
- 不同硬件和推理引擎对量化格式有什么限制。

## Speculative Decoding

Speculative Decoding 用一个更快的草稿模型或额外预测头先猜多个 token，再由目标模型验证，从而减少严格逐 token 解码带来的串行等待。

本节后续重点回答：

- draft、verify、acceptance rate 是什么。
- 为什么接受率、草稿模型速度和目标模型 batch 形态共同决定收益。
- Medusa、EAGLE、ngram speculation 等变体解决什么问题。

## Prefill/Decode 分离部署

Prefill/Decode 分离部署把 Prefill 和 Decode 放到不同 GPU 池或不同服务角色中，避免计算密集型 Prefill 与带宽敏感型 Decode 混在一起互相干扰。

本节后续重点回答：

- 分离部署为什么可能提高 goodput 和 SLO 达成率。
- KV Cache 如何从 Prefill worker 传给 Decode worker。
- 分离带来的网络传输、调度复杂度和容量规划问题。

## MoE 模型推理优化

MoE 模型每个 token 只激活部分专家，但系统上会引入专家路由、负载不均、跨卡通信和专家权重放置问题。MoE 推理优化的关键不只是算力，还包括通信和调度。

本节后续重点回答：

- expert parallel、routing、dispatch、combine 的系统代价是什么。
- 热门专家和冷门专家如何导致负载不均。
- MoE 推理中显存、通信和 batch 形态如何影响吞吐。

## 单机推理服务架构

单机推理服务关注一台服务器内的完整执行链路，包括模型加载、tokenizer、请求队列、scheduler、GPU executor、streaming server、metrics 和健康检查。

本节后续重点回答：

- 一个推理服务进程通常由哪些模块组成。
- CPU scheduler 与 GPU executor 如何协作。
- 单机服务如何处理并发、超时、取消请求和显存保护。

## 多机分布式推理

当模型过大、吞吐要求过高或上下文过长时，推理需要扩展到多 GPU、多节点。常见方式包括 tensor parallel、pipeline parallel、expert parallel、data parallel 和分离式 serving。

本节后续重点回答：

- 多 GPU 推理为什么会引入通信瓶颈。
- 不同并行方式如何影响 latency、throughput 和显存。
- 多机推理中网络、调度、失败恢复和容量规划如何处理。

## 缓存体系

推理系统里的缓存不只有 KV Cache。实际服务还可能包含 query cache、embedding cache、retrieval cache、prefix cache、tool result cache、model artifact cache 和 response cache。

本节后续重点回答：

- 每类缓存缓存的是什么，命中后节省哪段开销。
- 缓存命中率、过期策略、一致性和安全隔离如何设计。
- RAG、Agent 和长上下文服务为什么更依赖缓存体系。

## Benchmark 方法

Benchmark 方法关注如何设计实验，让性能数字可解释、可复现、可比较。推理 Benchmark 需要明确模型、硬件、精度、input length、output length、并发、请求分布和 SLO。

本节后续重点回答：

- synthetic workload 和 production trace 各有什么问题。
- 如何区分 prefill-heavy、decode-heavy、mixed workload。
- 为什么必须同时报告 latency、throughput、显存和成本。

## vLLM

vLLM 是现代开源 LLM serving 的重要案例，适合用来学习 PagedAttention、continuous batching、prefix caching、quantization、speculative decoding 和 OpenAI-compatible serving。

本节后续重点回答：

- vLLM 的核心设计如何围绕 KV Cache 和调度展开。
- vLLM 适合哪些在线服务场景，限制在哪里。
- 如何用 vLLM 做基准测试、服务部署和性能调优。

## TensorRT-LLM

TensorRT-LLM 是 NVIDIA 面向高性能 LLM 推理的优化栈，覆盖引擎构建、量化、并行、KV Cache、guided decoding、speculative decoding 和多种硬件优化。

本节后续重点回答：

- TensorRT-LLM 与通用 PyTorch runtime 的差异。
- engine build、kernel optimization、CUDA Graph 和量化如何影响性能。
- 在 NVIDIA GPU 上如何做高吞吐和低延迟部署。

## SGLang

SGLang 同时关注高性能 serving runtime 和结构化生成。它适合学习 RadixAttention、prefix reuse、structured outputs、continuous batching、prefill-decode disaggregation 和多模态 serving。

本节后续重点回答：

- SGLang 如何把生成程序和 runtime 优化结合起来。
- RadixAttention 与 prefix cache 的关系是什么。
- 结构化输出、工具调用和多轮生成如何影响推理系统。

## RAG / Agent 推理负载

RAG / Agent 不是单次模型调用，而是检索、rerank、上下文拼接、工具调用、多轮规划和多次 LLM 调用组成的复合 workload。它会放大尾延迟、缓存复杂度和失败概率。

详见：[RAG 与 Agent 推理负载](rag-agent-workloads.md)

本节后续重点回答：

- RAG / Agent 的端到端延迟如何拆解。
- 检索、工具调用和多轮 LLM 调用如何影响系统容量。
- 如何同时评估质量、延迟、成本和可靠性。

## Benchmark 方法与性能剖析

性能剖析是 Benchmark 之后的解释过程。Benchmark 告诉我们系统表现如何，profiling、trace、kernel timeline、memory profile 和 queue analysis 才能解释为什么。

本节后续重点回答：

- 如何从端到端指标定位到队列、CPU、GPU、网络或缓存瓶颈。
- 如何用 profiling 证据解释 TTFT、TPOT 和尾延迟。
- 如何把实验结果沉淀成容量模型和优化决策。
