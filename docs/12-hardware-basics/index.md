---
title: 硬件基础
domain: hardware
doc_type: overview
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-25
sources:
  - https://gitcode.com/cann/cannbot-skills
  - https://gitcode.com/cann/cannbot-skills/blob/master/README.md
---

# 硬件基础

本章把“具体硬件平台怎么认识、怎么适配、怎么沉淀成 AI 可用经验”单独拿出来。它和 [AI 加速器与计算架构](../06-accelerators-architecture/index.md) 的关系是：

- `AI 加速器与计算架构` 讲通用原理：算力、带宽、存储层次、互连、功耗、workload mapping。
- `硬件基础` 讲具体平台：型号怎么叫、软件栈怎么识别硬件、算子和模型服务适配时该收集哪些证据、怎么把经验写成 skill。

第一版聚焦 NPU 和昇腾 Ascend/CANN 生态，只围绕服务器、训练和推理加速场景组织内容，不展开端侧移动芯片线索。

## 本章结构

| 主题 | 解决的问题 |
| --- | --- |
| [NPU 基础概念](npu-basics.md) | NPU 是什么，为什么 AI 系统工程师需要理解 NPU 的执行模型和软件栈。 |
| [昇腾 NPU 型号与架构映射](ascend-npu-models.md) | 产品名、芯片型号、SocVersion、NpuArch、编译宏之间是什么关系。 |
| [Ascend 910 系列](ascend-910-series.md) | 910/910B/910_93 这类服务器训练和推理平台应该重点关注什么。 |
| [Ascend 950 系列](ascend-950-series.md) | 950PR/950DT 这类新平台的公开线索、学习重点和验证边界。 |
| [CANN 软件栈与开发入口](cann-stack.md) | CANN、Ascend C、torch_npu、算子开发、模型推理优化和 profiling 如何连接。 |
| [NPU 相关 AI Skills 样例](ai-skills-sample.md) | 什么内容适合写成 AI skill，以及本仓库如何给后续工作打样。 |

## 学习顺序

1. 先读 [NPU 基础概念](npu-basics.md)，建立“硬件不是孤立芯片，而是硬件、runtime、compiler、framework 和 workload 的组合”的视角。
2. 再读 [昇腾 NPU 型号与架构映射](ascend-npu-models.md)，理解为什么不能只说“910B”或“950”，而要同时记录 CANN 版本、SocVersion、NpuArch 和实际设备信息。
3. 根据手头平台选择 [Ascend 910 系列](ascend-910-series.md) 或 [Ascend 950 系列](ascend-950-series.md)。
4. 如果要做迁移、算子、性能优化或问题诊断，继续读 [CANN 软件栈与开发入口](cann-stack.md)。
5. 如果要让 AI 以后能复用团队经验，读 [NPU 相关 AI Skills 样例](ai-skills-sample.md)，并参考 `skills/` 目录中的样例写法。

## 写作边界

硬件平台文档要谨慎区分三类信息：

- `公开资料`：产品发布、路线图、媒体报道、开源仓库 README，适合做背景，不适合直接当作性能承诺。
- `软件栈可验证信息`：CANN、driver、runtime、framework、platform config、profiling 输出，适合写入工程判断。
- `本地实验结论`：benchmark、profiler、错误复现、调优记录，必须带上硬件、软件版本和 workload。

后续如果补充内部实验或实际部署经验，建议先写成 benchmark report、failure case 或 ADR，再抽象成可被 AI 调用的 skill。
