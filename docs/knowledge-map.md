---
title: AI 知识地图
status: reviewed
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# AI 知识地图

这张地图面向 AI 方向课题组新生，按“基础课 -> 经典模型 -> 数据与实验 -> 系统实现 -> 论文复现 -> 研究沉淀”的训练路径组织。图中带编号的模块可以点击跳转到对应章节。

## 总览思维导图

```mermaid
flowchart LR
  KG(("AI Knowledge Graph<br/>研究生入门 / 技术研究 / AI 检索"))

  KG --> S1["理论与模型基础<br/>先读懂论文语言"]
  KG --> S2["数据、训练与评测<br/>把实验做扎实"]
  KG --> S3["推理、检索与智能体<br/>研究模型交互能力"]
  KG --> S4["系统软件与计算架构<br/>理解 AI 如何跑起来"]
  KG --> S5["可信 AI 与研究工作流<br/>复现、记录、沉淀"]

  S1 --> M01["01 数学与机器学习基础"]
  S1 --> M02["02 模型架构与任务范式"]

  S2 --> M03["03 数据集、标注与基准"]
  S2 --> M04["04 训练算法、微调与对齐"]
  S2 --> M10["10 评测、复现与性能分析"]

  S3 --> M05["05 推理算法与高效服务"]
  S3 --> M06["06 检索增强、工具使用与智能体"]

  S4 --> M07["07 框架、编译器与运行时"]
  S4 --> M08["08 AI 计算架构与硬件基础"]
  S4 --> M09["09 分布式系统与实验平台"]

  S5 --> M11["11 可信、安全与 AI 治理"]
  S5 --> M12["12 论文复现、研究案例与决策记录"]
  S5 --> M13["13 知识组织与研究工作流"]
  S5 --> M99["99 模板与资源"]

  click KG "../" "打开首页"
  click M01 "../01-ai-basics/" "打开数学与机器学习基础"
  click M02 "../02-models-and-tasks/" "打开模型架构与任务范式"
  click M03 "../03-data-engineering/" "打开数据集、标注与基准"
  click M04 "../04-training-finetuning-alignment/" "打开训练算法、微调与对齐"
  click M05 "../05-inference-apps/" "打开推理算法与高效服务"
  click M06 "../06-prompt-rag-agents/" "打开检索增强、工具使用与智能体"
  click M07 "../07-ai-software-stack/" "打开框架、编译器与运行时"
  click M08 "../08-ai-compute-infra/" "打开 AI 计算架构与硬件基础"
  click M09 "../09-systems-mlops/" "打开分布式系统与实验平台"
  click M10 "../10-evaluation-benchmark-optimization/" "打开评测、复现与性能分析"
  click M11 "../11-safety-governance/" "打开可信、安全与 AI 治理"
  click M12 "../12-architecture-cases/" "打开论文复现、研究案例与决策记录"
  click M13 "../13-ai-indexing/" "打开知识组织与研究工作流"
  click M99 "../99-templates/knowledge-note/" "打开模板"

  classDef root fill:#111827,color:#ffffff,stroke:#111827,stroke-width:2px;
  classDef trunk fill:#eef2ff,color:#312e81,stroke:#4f46e5,stroke-width:1.2px;
  classDef module fill:#ffffff,color:#111827,stroke:#64748b,stroke-width:1.4px;
  class KG root;
  class S1,S2,S3,S4,S5 trunk;
  class M01,M02,M03,M04,M05,M06,M07,M08,M09,M10,M11,M12,M13,M99 module;
```

## 分支展开

### 理论与模型基础

```mermaid
flowchart LR
  S1["理论与模型基础"] --> M01["01 数学与机器学习基础"]
  S1 --> M02["02 模型架构与任务范式"]

  M01 --> A1["线性代数 / 概率统计 / 最优化"]
  M01 --> A2["统计学习 / 泛化 / 正则化"]
  M01 --> A3["神经网络 / 反向传播 / 表示学习"]

  M02 --> B1["任务形式化<br/>分类 / 检索 / 生成 / 多模态"]
  M02 --> B2["架构谱系<br/>CNN / Transformer / Diffusion / MoE"]
  M02 --> B3["归纳偏置 / 能力边界 / 失效模式"]

  click M01 "../01-ai-basics/" "打开数学与机器学习基础"
  click M02 "../02-models-and-tasks/" "打开模型架构与任务范式"
```

### 数据、训练与评测

```mermaid
flowchart LR
  S2["数据、训练与评测"] --> M03["03 数据集、标注与基准"]
  S2 --> M04["04 训练算法、微调与对齐"]
  S2 --> M10["10 评测、复现与性能分析"]

  M03 --> C1["数据集构建 / 标注协议 / 数据卡"]
  M03 --> C2["数据泄漏 / 污染检测 / 分布偏移"]
  M03 --> C3["Benchmark 设计 / 训练验证测试切分"]

  M04 --> D1["预训练 / SFT / 参数高效微调"]
  M04 --> D2["Optimizer / Schedule / 正则化"]
  M04 --> D3["RLHF / DPO / 偏好优化"]
  M04 --> D4["随机性控制 / Checkpoint / 可复现训练"]

  M10 --> E1["能力评测 / 消融实验 / 对照基线"]
  M10 --> E2["统计显著性 / 置信区间 / 误差分析"]
  M10 --> E3["Profiling / Roofline / 资源效率"]

  click M03 "../03-data-engineering/" "打开数据集、标注与基准"
  click M04 "../04-training-finetuning-alignment/" "打开训练算法、微调与对齐"
  click M10 "../10-evaluation-benchmark-optimization/" "打开评测、复现与性能分析"
```

### 推理、检索与智能体

```mermaid
flowchart LR
  S3["推理、检索与智能体"] --> M05["05 推理算法与高效服务"]
  S3 --> M06["06 检索增强、工具使用与智能体"]

  M05 --> F1["KV Cache / PagedAttention / Continuous Batching"]
  M05 --> F2["Speculative Decoding / 量化 / 蒸馏"]
  M05 --> F3["Latency / Throughput / TTFT / TPOT"]

  M06 --> G1["In-context Learning / CoT / ReAct"]
  M06 --> G2["Embedding / Hybrid Search / Rerank"]
  M06 --> G3["RAG 忠实度 / 引用 / 归因评测"]
  M06 --> G4["Tool Use / Planning / Agent Benchmark"]

  click M05 "../05-inference-apps/" "打开推理算法与高效服务"
  click M06 "../06-prompt-rag-agents/" "打开检索增强、工具使用与智能体"
```

### 系统软件与计算架构

```mermaid
flowchart LR
  S4["系统软件与计算架构"] --> M07["07 框架、编译器与运行时"]
  S4 --> M08["08 AI 计算架构与硬件基础"]
  S4 --> M09["09 分布式系统与实验平台"]

  M07 --> H1["PyTorch / JAX / ONNX"]
  M07 --> H2["Triton / CUDA / ROCm / Kernel"]
  M07 --> H3["编译器 / Runtime / 自动调优"]

  M08 --> I1["GPU / NPU / ASIC / FPGA"]
  M08 --> I2["HBM / Cache / NUMA / Memory Wall"]
  M08 --> I3["PCIe / CXL / NVLink / NoC"]
  M08 --> I4["算术强度 / 带宽瓶颈 / 能效"]

  M09 --> J1["数据并行 / 张量并行 / 流水线并行"]
  M09 --> J2["AllReduce / 通信重叠 / 容错"]
  M09 --> J3["Slurm / Kubernetes / Ray / 实验追踪"]

  click M07 "../07-ai-software-stack/" "打开框架、编译器与运行时"
  click M08 "../08-ai-compute-infra/" "打开 AI 计算架构与硬件基础"
  click M09 "../09-systems-mlops/" "打开分布式系统与实验平台"
```

### 可信 AI 与研究工作流

```mermaid
flowchart LR
  S5["可信 AI 与研究工作流"] --> M11["11 可信、安全与 AI 治理"]
  S5 --> M12["12 论文复现、研究案例与决策记录"]
  S5 --> M13["13 知识组织与研究工作流"]
  S5 --> M99["99 模板与资源"]

  M11 --> K1["对抗样本 / 投毒 / 后门 / 模型窃取"]
  M11 --> K2["幻觉 / 偏见 / 校准 / 可解释性"]
  M11 --> K3["隐私保护 / 许可 / 模型卡"]

  M12 --> L1["论文精读 / 代码走读 / 复现报告"]
  M12 --> L2["研究决策 / 实验假设 / 负结果"]
  M12 --> L3["实验故障 / 失败模式 / 复盘"]

  M13 --> N1["Front Matter / 标签 / 来源路径"]
  M13 --> N2["向量索引 / 知识图谱 / 引用溯源"]
  M13 --> N3["AI Skills / Agent 工具说明"]

  M99 --> T1["知识点模板"]
  M99 --> T2["研究决策模板"]
  M99 --> T3["Benchmark 报告模板"]

  click M11 "../11-safety-governance/" "打开可信、安全与 AI 治理"
  click M12 "../12-architecture-cases/" "打开论文复现、研究案例与决策记录"
  click M13 "../13-ai-indexing/" "打开知识组织与研究工作流"
  click M99 "../99-templates/knowledge-note/" "打开模板"
```

## 地图逻辑

| 主线 | 组织逻辑 | 对应模块 |
| --- | --- | --- |
| 理论与模型基础 | 先补足数学、统计学习和深度学习基础，再理解任务形式化和模型架构谱系。 | [01 数学与机器学习基础](01-ai-basics/index.md)、[02 模型架构与任务范式](02-models-and-tasks/index.md) |
| 数据、训练与评测 | 数据定义实验边界，训练算法产生模型能力，评测与复现判断结论是否可靠。 | [03 数据集、标注与基准](03-data-engineering/index.md)、[04 训练算法、微调与对齐](04-training-finetuning-alignment/index.md)、[10 评测、复现与性能分析](10-evaluation-benchmark-optimization/index.md) |
| 推理、检索与智能体 | 研究模型在推理阶段的效率、外部知识接入、工具使用和长程任务能力。 | [05 推理算法与高效服务](05-inference-apps/index.md)、[06 检索增强、工具使用与智能体](06-prompt-rag-agents/index.md) |
| 系统软件与计算架构 | 理解模型如何被框架、编译器、运行时、加速器和分布式系统共同执行。 | [07 框架、编译器与运行时](07-ai-software-stack/index.md)、[08 AI 计算架构与硬件基础](08-ai-compute-infra/index.md)、[09 分布式系统与实验平台](09-systems-mlops/index.md) |
| 可信 AI 与研究工作流 | 控制研究风险，沉淀论文复现、实验决策、失败经验和 AI 可读知识。 | [11 可信、安全与 AI 治理](11-safety-governance/index.md)、[12 论文复现、研究案例与决策记录](12-architecture-cases/index.md)、[13 知识组织与研究工作流](13-ai-indexing/index.md) |

## 按研究任务导航

| 研究生当前任务 | 优先阅读 |
| --- | --- |
| 刚入组，需要补基础 | [01 数学与机器学习基础](01-ai-basics/index.md) -> [02 模型架构与任务范式](02-models-and-tasks/index.md) |
| 准备读一篇模型论文 | [02 模型架构与任务范式](02-models-and-tasks/index.md) -> [03 数据集、标注与基准](03-data-engineering/index.md) -> [10 评测、复现与性能分析](10-evaluation-benchmark-optimization/index.md) |
| 准备复现训练或微调论文 | [03 数据集、标注与基准](03-data-engineering/index.md) -> [04 训练算法、微调与对齐](04-training-finetuning-alignment/index.md) -> [12 论文复现、研究案例与决策记录](12-architecture-cases/index.md) |
| 做 LLM 推理、RAG 或 Agent 方向 | [05 推理算法与高效服务](05-inference-apps/index.md) -> [06 检索增强、工具使用与智能体](06-prompt-rag-agents/index.md) -> [10 评测、复现与性能分析](10-evaluation-benchmark-optimization/index.md) |
| 做 AI 系统、编译器或硬件方向 | [07 框架、编译器与运行时](07-ai-software-stack/index.md) -> [08 AI 计算架构与硬件基础](08-ai-compute-infra/index.md) -> [09 分布式系统与实验平台](09-systems-mlops/index.md) |
| 做可信 AI 或安全方向 | [11 可信、安全与 AI 治理](11-safety-governance/index.md) -> [10 评测、复现与性能分析](10-evaluation-benchmark-optimization/index.md) -> [12 论文复现、研究案例与决策记录](12-architecture-cases/index.md) |
| 准备沉淀组会、论文笔记或实验记录 | [13 知识组织与研究工作流](13-ai-indexing/index.md) -> [知识点模板](99-templates/knowledge-note.md) -> [Benchmark 报告模板](99-templates/benchmark-report.md) |

## 模块关系

| 模块 | 上游依赖 | 主要产出 |
| --- | --- | --- |
| 01 数学与机器学习基础 | 无 | 术语、公式、基础理论和论文阅读前置知识 |
| 02 模型架构与任务范式 | 01 | 任务形式化、模型谱系、架构理解和开放问题 |
| 03 数据集、标注与基准 | 02 | 数据卡、标注协议、benchmark 边界和数据风险 |
| 04 训练算法、微调与对齐 | 01、02、03 | 训练方案、优化配置、对齐方法和复现实验 |
| 05 推理算法与高效服务 | 02、04、07、08 | 推理算法、服务运行时、性能指标和瓶颈分析 |
| 06 检索增强、工具使用与智能体 | 02、05、10、13 | RAG、Tool Use、Agent 工作流和评测协议 |
| 07 框架、编译器与运行时 | 01、04、05 | Kernel、编译优化、Runtime 和系统实现知识 |
| 08 AI 计算架构与硬件基础 | 04、05、07 | 计算、内存、互连、能效和体系结构分析 |
| 09 分布式系统与实验平台 | 04、07、08 | 并行训练、调度、容错、实验追踪和复现平台 |
| 10 评测、复现与性能分析 | 02、03、04、05、06、08、09 | 指标体系、消融实验、复现协议和性能诊断 |
| 11 可信、安全与 AI 治理 | 03、05、06、10 | 威胁模型、安全评测、隐私与许可边界 |
| 12 论文复现、研究案例与决策记录 | 全部模块 | 论文笔记、复现报告、负结果和研究决策 |
| 13 知识组织与研究工作流 | 全部模块 | 元数据、索引、引用溯源、AI-readable skills |
