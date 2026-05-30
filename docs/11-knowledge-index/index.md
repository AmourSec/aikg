---
title: 知识组织、模板与 AI 可读索引
domain: knowledge-index
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 知识组织、模板与 AI 可读索引

本目录定义知识库如何被人阅读，也如何被 AI 检索、引用和转化为可执行的技能。AI Infra 知识高度依赖环境、指标和上下文，因此元数据必须足够结构化。

## 建议主题

- front matter 字段、标签体系、状态流转
- workload、hardware、software、metrics、source 的统一写法
- 论文笔记、Benchmark 报告、技术决策、故障复盘模板
- 引用溯源、版本记录、实验数据链接
- 向量索引、知识图谱、实体关系、AI skills
- 面向 agent 的任务说明、约束、输入输出格式和验证步骤

## 元数据字段

| 字段 | 说明 |
| --- | --- |
| title | 文档标题 |
| domain | 所属技术域 |
| workload | 相关 workload 或模型形态 |
| system_layer | workload / runtime / kernel / accelerator / cluster / measurement |
| hardware | GPU、NPU、CPU、网络、存储等环境 |
| software | framework、runtime、driver、compiler、library version |
| metrics | 使用的指标 |
| source | 论文、代码、实验记录或故障单 |
| status | draft / reviewed / verified / deprecated |

## 关键问题

- AI 是否能从文档中知道结论适用的 workload 和环境。
- 性能结论是否有指标、实验和来源支撑。
- 文档之间是否能通过标签、实体和链接形成可检索网络。
- 模板是否能降低新同学写复现实验和故障复盘的成本。
