---
title: 数据与输入路径
domain: data-paths
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 数据与输入路径

本页不讨论数据集如何提升任务指标，而关注数据如何进入训练和推理系统，以及数据路径如何成为性能瓶颈。

## 建议主题

- Tokenization、packing、padding、truncation
- dynamic batching、bucketing、sequence packing
- 数据格式：JSONL、Parquet、Arrow、WebDataset、二进制 shard
- 训练数据读取、shuffle、prefetch、cache、NUMA 亲和性
- 在线推理请求解析、输入校验、prompt 模板、上下文拼接
- Embedding、检索、rerank、工具调用的额外数据路径
- 对象存储、本地 NVMe、共享文件系统、数据缓存

## 关键问题

- 数据读取是否让 GPU 等待。
- Padding 和动态 shape 是否浪费计算。
- tokenizer、preprocess、postprocess 是否在 CPU 侧成为瓶颈。
- 数据分片和 shuffle 是否影响可复现性和训练吞吐。
- 推理服务中上下文拼接、检索和工具调用是否显著增加尾延迟。
