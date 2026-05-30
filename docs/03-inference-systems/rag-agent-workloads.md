---
title: RAG 与 Agent 推理负载
domain: rag-agent-workloads
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# RAG 与 Agent 推理负载

本页把 RAG 和 Agent 当作推理系统的复合 workload，而不是提示词技巧专题。重点是检索、rerank、工具调用、多轮规划和模型调用如何改变延迟、吞吐、成本和可靠性。

## 建议主题

- Embedding service、vector search、hybrid search、rerank
- chunking、context packing、引用拼接、上下文窗口占用
- tool calling、function execution、sandbox、timeout、retry
- multi-step agent、planner/executor、并行工具调用
- cache：query cache、embedding cache、retrieval cache、tool result cache
- end-to-end latency breakdown、fan-out、tail amplification
- RAG/Agent Benchmark、faithfulness、citation、tool success rate

## 关键问题

- 检索、rerank、LLM prefill、decode、工具调用分别贡献多少延迟。
- Agent 多步调用是否放大尾延迟和失败概率。
- 缓存命中率、召回质量和上下文长度之间如何取舍。
- 工具失败、超时、重复调用和权限隔离如何处理。
- 评测是否同时覆盖质量、延迟、成本和稳定性。
