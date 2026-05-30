---
title: AI 知识地图
status: reviewed
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# AI 知识地图

这张地图按“学习理解 -> 模型生产 -> 应用交付 -> 系统支撑 -> 经验复用”的闭环组织。先看总览图建立方向，再看每条主线的分支展开；图中带编号的模块可以点击跳转到对应章节。

## 总览思维导图

```mermaid
flowchart LR
  KG(("AI Knowledge Graph<br/>学习 / 工程 / 知识复用"))

  KG --> S1["学习理解<br/>先建立共同语言"]
  KG --> S2["模型生产<br/>从数据到可用模型"]
  KG --> S3["应用交付<br/>把模型变成产品能力"]
  KG --> S4["系统支撑<br/>稳定、高效、可扩展"]
  KG --> S5["风险与资产闭环<br/>治理、复盘、AI 可读"]

  S1 --> M01["01 AI 基础理论"]
  S1 --> M02["02 模型与任务"]

  S2 --> M03["03 数据与数据工程"]
  S2 --> M04["04 训练、微调与对齐"]
  S2 --> M10["10 评测、Benchmark 与优化"]

  S3 --> M05["05 推理与应用构建"]
  S3 --> M06["06 Prompt、RAG 与 Agents"]

  S4 --> M07["07 AI 软件栈"]
  S4 --> M08["08 AI 计算与基础设施"]
  S4 --> M09["09 系统、集群与 MLOps"]

  S5 --> M11["11 安全、治理与 Responsible AI"]
  S5 --> M12["12 架构案例、ADR 与故障复盘"]
  S5 --> M13["13 AI 可读知识层"]
  S5 --> M99["99 模板与资源"]

  click KG "../" "打开首页"
  click M01 "../01-ai-basics/" "打开 AI 基础理论"
  click M02 "../02-models-and-tasks/" "打开模型与任务"
  click M03 "../03-data-engineering/" "打开数据与数据工程"
  click M04 "../04-training-finetuning-alignment/" "打开训练、微调与对齐"
  click M05 "../05-inference-apps/" "打开推理与应用构建"
  click M06 "../06-prompt-rag-agents/" "打开 Prompt、RAG 与 Agents"
  click M07 "../07-ai-software-stack/" "打开 AI 软件栈"
  click M08 "../08-ai-compute-infra/" "打开 AI 计算与基础设施"
  click M09 "../09-systems-mlops/" "打开系统、集群与 MLOps"
  click M10 "../10-evaluation-benchmark-optimization/" "打开评测、Benchmark 与优化"
  click M11 "../11-safety-governance/" "打开安全、治理与 Responsible AI"
  click M12 "../12-architecture-cases/" "打开架构案例、ADR 与故障复盘"
  click M13 "../13-ai-indexing/" "打开 AI 可读知识层"
  click M99 "../99-templates/knowledge-note/" "打开模板"

  classDef root fill:#111827,color:#ffffff,stroke:#111827,stroke-width:2px;
  classDef trunk fill:#eef2ff,color:#312e81,stroke:#4f46e5,stroke-width:1.2px;
  classDef module fill:#ffffff,color:#111827,stroke:#64748b,stroke-width:1.4px;
  class KG root;
  class S1,S2,S3,S4,S5 trunk;
  class M01,M02,M03,M04,M05,M06,M07,M08,M09,M10,M11,M12,M13,M99 module;
```

## 分支展开

### 学习理解

```mermaid
flowchart LR
  S1["学习理解"] --> M01["01 AI 基础理论"]
  S1 --> M02["02 模型与任务"]

  M01 --> A1["数学 / ML / DL"]
  M01 --> A2["Attention / Transformer / LLM"]
  M01 --> A3["训练、推理、压缩基本概念"]

  M02 --> B1["任务范式<br/>分类 / 检索 / 生成 / 多模态"]
  M02 --> B2["模型谱系<br/>LLM / VLM / Diffusion / MoE"]
  M02 --> B3["能力边界与选型"]

  click M01 "../01-ai-basics/" "打开 AI 基础理论"
  click M02 "../02-models-and-tasks/" "打开模型与任务"
```

### 模型生产

```mermaid
flowchart LR
  S2["模型生产"] --> M03["03 数据与数据工程"]
  S2 --> M04["04 训练、微调与对齐"]
  S2 --> M10["10 评测、Benchmark 与优化"]

  M03 --> C1["采集 / 清洗 / 去重 / 脱敏"]
  M03 --> C2["标注 / 数据质量 / 数据泄漏"]
  M03 --> C3["Tokenization / Synthetic Data / 版本管理"]

  M04 --> D1["预训练 / 继续预训练 / SFT"]
  M04 --> D2["LoRA / QLoRA / Adapter"]
  M04 --> D3["RLHF / DPO / 偏好优化"]
  M04 --> D4["分布式训练 / 稳定性 / 可复现"]

  M10 --> E1["模型能力评测"]
  M10 --> E2["训练性能 / 推理性能"]
  M10 --> E3["Profiling / Roofline / TCO"]

  click M03 "../03-data-engineering/" "打开数据与数据工程"
  click M04 "../04-training-finetuning-alignment/" "打开训练、微调与对齐"
  click M10 "../10-evaluation-benchmark-optimization/" "打开评测、Benchmark 与优化"
```

### 应用交付

```mermaid
flowchart LR
  S3["应用交付"] --> M05["05 推理与应用构建"]
  S3 --> M06["06 Prompt、RAG 与 Agents"]

  M05 --> F1["在线推理 / 批处理 / 流式输出"]
  M05 --> F2["KV Cache / Batching / Speculative Decoding"]
  M05 --> F3["API / 限流 / 缓存 / 重试"]
  M05 --> F4["结构化输出 / 工具调用 / 多模态应用"]

  M06 --> G1["Prompt / Context Engineering"]
  M06 --> G2["Embedding / Hybrid Search / Rerank"]
  M06 --> G3["RAG 切分 / 召回 / 引用 / 评测"]
  M06 --> G4["Tool Calling / MCP / Agent 记忆"]

  click M05 "../05-inference-apps/" "打开推理与应用构建"
  click M06 "../06-prompt-rag-agents/" "打开 Prompt、RAG 与 Agents"
```

### 系统支撑

```mermaid
flowchart LR
  S4["系统支撑"] --> M07["07 AI 软件栈"]
  S4 --> M08["08 AI 计算与基础设施"]
  S4 --> M09["09 系统、集群与 MLOps"]

  M07 --> H1["PyTorch / JAX / ONNX"]
  M07 --> H2["CUDA / ROCm / Triton / Kernel"]
  M07 --> H3["编译器 / Runtime / Serving"]

  M08 --> I1["CPU / GPU / NPU / ASIC / FPGA"]
  M08 --> I2["HBM / DDR / Cache / NUMA"]
  M08 --> I3["PCIe / CXL / NVLink / NoC"]
  M08 --> I4["能效 / 可靠性 / 容量规划"]

  M09 --> J1["Kubernetes / Slurm / 调度"]
  M09 --> J2["网络 / 存储 / 监控"]
  M09 --> J3["实验管理 / 模型注册 / CI-CD"]
  M09 --> J4["漂移检测 / 反馈闭环"]

  click M07 "../07-ai-software-stack/" "打开 AI 软件栈"
  click M08 "../08-ai-compute-infra/" "打开 AI 计算与基础设施"
  click M09 "../09-systems-mlops/" "打开系统、集群与 MLOps"
```

### 风险与资产闭环

```mermaid
flowchart LR
  S5["风险与资产闭环"] --> M11["11 安全、治理与 Responsible AI"]
  S5 --> M12["12 架构案例、ADR 与故障复盘"]
  S5 --> M13["13 AI 可读知识层"]
  S5 --> M99["99 模板与资源"]

  M11 --> K1["Prompt Injection / 越权工具调用"]
  M11 --> K2["幻觉 / 事实性 / 偏见"]
  M11 --> K3["Guardrails / Red Teaming"]
  M11 --> K4["数据许可 / 模型许可 / 审计"]

  M12 --> L1["端到端系统案例"]
  M12 --> L2["设计决策与取舍"]
  M12 --> L3["故障复盘 / 模式 / 反模式"]

  M13 --> N1["Front Matter / 标签 / 来源路径"]
  M13 --> N2["向量索引 / 知识图谱"]
  M13 --> N3["AI Skills / Agent 工具说明"]

  M99 --> T1["知识点模板"]
  M99 --> T2["ADR 模板"]
  M99 --> T3["Benchmark 模板"]

  click M11 "../11-safety-governance/" "打开安全、治理与 Responsible AI"
  click M12 "../12-architecture-cases/" "打开架构案例、ADR 与故障复盘"
  click M13 "../13-ai-indexing/" "打开 AI 可读知识层"
  click M99 "../99-templates/knowledge-note/" "打开模板"
```

## 地图逻辑

| 主线 | 组织逻辑 | 对应模块 |
| --- | --- | --- |
| 学习理解 | 先统一基础概念，再理解模型类型、任务边界和选型方式。 | [01 AI 基础理论](01-ai-basics/index.md)、[02 模型与任务](02-models-and-tasks/index.md) |
| 模型生产 | 数据决定上限，训练和对齐决定能力形态，评测负责判断是否真的可用。 | [03 数据与数据工程](03-data-engineering/index.md)、[04 训练、微调与对齐](04-training-finetuning-alignment/index.md)、[10 评测、Benchmark 与优化](10-evaluation-benchmark-optimization/index.md) |
| 应用交付 | 推理服务负责把模型稳定暴露出来，Prompt、RAG 和 Agents 负责把模型接入业务知识、工具和流程。 | [05 推理与应用构建](05-inference-apps/index.md)、[06 Prompt、RAG 与 Agents](06-prompt-rag-agents/index.md) |
| 系统支撑 | 软件栈、计算基础设施、集群与 MLOps 决定训练和推理能否规模化、低成本、可观测地运行。 | [07 AI 软件栈](07-ai-software-stack/index.md)、[08 AI 计算与基础设施](08-ai-compute-infra/index.md)、[09 系统、集群与 MLOps](09-systems-mlops/index.md) |
| 风险与资产闭环 | 安全治理控制风险，架构案例和 ADR 沉淀判断依据，AI 可读层让知识能被检索、引用和复用。 | [11 安全、治理与 Responsible AI](11-safety-governance/index.md)、[12 架构案例、ADR 与故障复盘](12-architecture-cases/index.md)、[13 AI 可读知识层](13-ai-indexing/index.md) |

## 按问题导航

| 我想解决的问题 | 优先阅读 |
| --- | --- |
| 我刚开始系统学习 AI | [01 AI 基础理论](01-ai-basics/index.md) -> [02 模型与任务](02-models-and-tasks/index.md) |
| 我想判断某个任务该用什么模型 | [02 模型与任务](02-models-and-tasks/index.md) -> [10 评测、Benchmark 与优化](10-evaluation-benchmark-optimization/index.md) |
| 我想建设训练或微调能力 | [03 数据与数据工程](03-data-engineering/index.md) -> [04 训练、微调与对齐](04-training-finetuning-alignment/index.md) -> [10 评测、Benchmark 与优化](10-evaluation-benchmark-optimization/index.md) |
| 我想做 RAG、Agent 或知识库应用 | [05 推理与应用构建](05-inference-apps/index.md) -> [06 Prompt、RAG 与 Agents](06-prompt-rag-agents/index.md) -> [13 AI 可读知识层](13-ai-indexing/index.md) |
| 我想把模型服务跑稳、跑快、跑便宜 | [05 推理与应用构建](05-inference-apps/index.md) -> [07 AI 软件栈](07-ai-software-stack/index.md) -> [08 AI 计算与基础设施](08-ai-compute-infra/index.md) -> [09 系统、集群与 MLOps](09-systems-mlops/index.md) |
| 我想分析性能瓶颈、算力效率和系统成本 | [07 AI 软件栈](07-ai-software-stack/index.md) -> [08 AI 计算与基础设施](08-ai-compute-infra/index.md) -> [10 评测、Benchmark 与优化](10-evaluation-benchmark-optimization/index.md) -> [12 架构案例、ADR 与故障复盘](12-architecture-cases/index.md) |
| 我想避免安全、许可、隐私和治理风险 | [11 安全、治理与 Responsible AI](11-safety-governance/index.md) -> [12 架构案例、ADR 与故障复盘](12-architecture-cases/index.md) |
| 我想让资料库更适合 AI 检索和引用 | [13 AI 可读知识层](13-ai-indexing/index.md) -> [知识点模板](99-templates/knowledge-note.md) -> [ADR 模板](99-templates/adr.md) |

## 模块关系

| 模块 | 上游依赖 | 主要产出 |
| --- | --- | --- |
| 01 AI 基础理论 | 无 | 统一概念、术语和基本原理 |
| 02 模型与任务 | 01 | 任务分类、模型选型、能力边界 |
| 03 数据与数据工程 | 02 | 可训练、可评测、可追溯的数据资产 |
| 04 训练、微调与对齐 | 01、02、03 | 可用模型、对齐策略、训练经验 |
| 05 推理与应用构建 | 02、04、07、08 | 服务接口、推理链路、应用能力 |
| 06 Prompt、RAG 与 Agents | 02、05、13 | 上下文工程、检索增强、工具化工作流 |
| 07 AI 软件栈 | 01、04、05 | 框架、Runtime、Kernel、Serving 能力 |
| 08 AI 计算与基础设施 | 04、05、07 | 算力、内存、互连、容量和能效模型 |
| 09 系统、集群与 MLOps | 05、07、08 | 部署、调度、监控、发布和反馈闭环 |
| 10 评测、Benchmark 与优化 | 02、03、04、05、08、09 | 能力评测、性能评测、瓶颈定位、成本判断 |
| 11 安全、治理与 Responsible AI | 03、05、06、13 | 风险边界、治理策略、安全评测 |
| 12 架构案例、ADR 与故障复盘 | 05、07、08、09、10、11 | 可复用设计判断、失败模式、反模式 |
| 13 AI 可读知识层 | 全部模块 | 元数据、索引、实体关系、AI skills |
