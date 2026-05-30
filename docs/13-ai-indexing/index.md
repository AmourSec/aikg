---
title: 知识组织与研究工作流
domain: ai-indexing
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 知识组织与研究工作流

本目录沉淀研究知识的组织方式，使论文笔记、实验记录、代码分析和系统经验既适合人阅读，也适合 AI 检索、引用和复用。

## 基础要求

- 每篇文档保留 front matter 元数据。
- 文档标题、主题、来源、owner、状态、授权、更新时间必须明确。
- 论文笔记需要保留论文链接、代码链接、关键公式、实验设置和复现状态。
- 文档进入向量库或知识图谱前需要保留来源路径、提交版本和引用信息。
- AI 输出引用知识时需要给出来源路径、更新时间和适用范围。

## 推荐元数据字段

| 字段 | 含义 |
| --- | --- |
| `title` | 文档标题 |
| `domain` | 所属知识域 |
| `status` | draft / reviewed / deprecated |
| `source` | 论文、教材、代码、实验记录或链接 |
| `owner` | 维护者 |
| `updated` | 最近更新时间 |
| `license` | 授权方式 |
