---
title: AI 可读知识层
domain: ai-indexing
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# AI 可读知识层

本目录定义知识库如何被 AI 检索、引用和推理。

## 基础要求

- 每篇文档保留 front matter 元数据。
- 文档标题、标签、owner、状态、授权、更新时间必须明确。
- 文档进入向量库或知识图谱前需要保留来源路径和版本信息。
- AI 输出引用知识时需要给出来源路径和更新时间。

## 推荐元数据字段

| 字段 | 说明 |
| --- | --- |
| `title` | 文档标题 |
| `domain` | 知识领域 |
| `layer` | 计算、软件、系统、Benchmark、架构等层级 |
| `product` | 适用产品或平台 |
| `owner` | 负责人或团队 |
| `status` | draft、reviewed、deprecated |
| `license` | 内容授权，例如 CC-BY-4.0 |
| `updated` | 最后更新日期 |
| `related` | 相关知识点 |
