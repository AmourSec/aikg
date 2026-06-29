---
title: 硬件基础
domain: hardware
doc_type: overview
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-29
sources:
  - https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
  - https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
  - https://gitcode.com/cann/cannbot-skills
  - https://gitcode.com/cann/cannbot-skills/blob/master/README.md
---

# 硬件基础

本章把“具体硬件平台怎么认识、怎么适配、怎么沉淀成 AI 可用经验”单独拿出来。它和 [AI 加速器与计算架构](../06-accelerators-architecture/index.md) 的边界是：

- `AI 加速器与计算架构` 讲通用原理：算力、带宽、存储层次、互连、功耗、workload mapping。
- `硬件基础` 讲具体平台入口：GPU/NPU 怎么读架构图，昇腾平台怎么识别型号和架构，CANN 软件栈怎么进入，以及硬件适配经验怎么沉淀给 AI。

本章围绕服务器、训练和推理加速场景组织内容。读者应该先建立 GPU/NPU 的硬件直觉，再进入昇腾 Ascend/CANN 生态；不展开端侧移动芯片线索。

## 本章结构

### 通用硬件架构

| 主题 | 解决的问题 |
| --- | --- |
| [GPU 架构基础](gpu-architecture-basics.md) | 结合 NVIDIA 官方架构图理解 GPU、host/device、grid/block/thread、SM、warp、Tensor Core、显存层次和常见操作。 |
| [NPU 架构基础](npu-basics.md) | 结合 Ascend/CANN 架构图理解 NPU、AI Core、Cube/Vector、片上存储、DataCopy、Tiling 和软件栈边界。 |
| [GPU 与 NPU 异同点](gpu-npu-comparison.md) | GPU/NPU 在执行模型、软件栈、内存、算子覆盖、训练推理和迁移适配上有什么相同和不同。 |

### 昇腾平台与 CANN

| 主题 | 解决的问题 |
| --- | --- |
| [Ascend 型号、SocVersion 与 NpuArch](ascend-npu-models.md) | 产品系列、芯片型号、SocVersion、NpuArch、`__NPU_ARCH__` 和 archXX 之间是什么关系。 |
| [Ascend 910 系列平台要点](ascend-910-series.md) | 910/910B/910_93 这类服务器训练和推理平台应该重点关注什么。 |
| [Ascend 950 系列平台要点](ascend-950-series.md) | 950PR/950DT 这类新平台的公开线索、学习重点和验证边界。 |
| [Ascend/CANN 软件栈与开发入口](cann-stack.md) | CANN、Ascend C、torch_npu、算子开发、模型推理优化和 profiling 如何连接。 |

### 硬件知识沉淀

| 主题 | 解决的问题 |
| --- | --- |
| [硬件适配 AI Skills 样例](ai-skills-sample.md) | 什么内容适合写成 AI skill，以及本仓库如何给后续硬件适配工作打样。 |

## 学习顺序

1. 先读 [GPU 架构基础](gpu-architecture-basics.md) 和 [NPU 架构基础](npu-basics.md)，建立“硬件不是孤立芯片，而是硬件、runtime、compiler、framework 和 workload 的组合”的视角。
2. 再读 [GPU 与 NPU 异同点](gpu-npu-comparison.md)，理解哪些 GPU 经验可以迁移，哪些必须重新验证。
3. 如果关注昇腾平台，继续读 [Ascend 型号、SocVersion 与 NpuArch](ascend-npu-models.md)，先把产品名、软件识别名、架构号和编译宏分清楚。
4. 根据手头平台选择 [Ascend 910 系列平台要点](ascend-910-series.md) 或 [Ascend 950 系列平台要点](ascend-950-series.md)。
5. 如果要做迁移、算子、性能优化或问题诊断，继续读 [Ascend/CANN 软件栈与开发入口](cann-stack.md)。
6. 如果要让 AI 以后能复用团队经验，读 [硬件适配 AI Skills 样例](ai-skills-sample.md)，并参考 `skills/` 目录中的样例写法。

## 写作边界

硬件平台文档要谨慎区分三类信息：

- `公开资料`：产品发布、路线图、媒体报道、开源仓库 README，适合做背景，不适合直接当作性能承诺。
- `软件栈可验证信息`：CANN、driver、runtime、framework、platform config、profiling 输出，适合写入工程判断。
- `本地实验结论`：benchmark、profiler、错误复现、调优记录，必须带上硬件、软件版本和 workload。

后续如果补充内部实验或实际部署经验，建议先写成 benchmark report、failure case 或 ADR，再抽象成可被 AI 调用的 skill。
