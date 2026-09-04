---
title: 首页
status: reviewed
owner: maintainers
license: CC-BY-4.0
updated: 2026-09-04
---

# AI Knowledge Graph

这是一个面向 AI Systems、AI Infrastructure 和高效 AI 计算方向的开放知识库。它不以提升模型任务指标为主线，而以“让 AI 负载跑得更快、更省、更稳、更可复现”为主线，沉淀推理系统、分布式训练、Kernel 与编译、加速器架构、集群基础设施、性能分析和系统论文复现。

## 这个仓库能为你做什么

本仓库不只是一个文章库，它同时服务人和 AI，有两个主要功能；其中 skills 自动路由还预留了一个默认关闭的远程检索扩展点。第一次来可以先看这一节，再决定往哪里走。

| 功能 | 在哪里 | 给谁用 |
| --- | --- | --- |
| AI 基础知识文章库 | `docs/`（渲染为 MkDocs 站点） | 人和 AI |
| Skills 库与自动技能路由 | `skills/` + `automatic-skill-routing/` | AI Agent |

### 功能一：AI 基础知识文章库

`docs/` 下是按学习顺序组织的 AI 系统与 Infra 文章，从 AI 基础概念一路讲到硬件、推理系统、训练系统、Kernel 优化和集群，渲染成 MkDocs 站点，并附带给 AI 用的索引文件。

- **怎么用**：人从[入门导读](01-getting-started/)开始按学习路径阅读即可；AI 助手则把仓库 GitHub 链接和入口索引 `https://amoursec.github.io/aikg/llms.txt` 一起给它，AI 会按索引检索文章并引用原文路径。
- **能做什么**：建立性能视角的共同语言；在写方案、调性能、排查问题时，作为可引用的背景知识来源。

### 功能二：Skills 库与自动技能路由（告诉 AI 仓库链接即可自动找到合适的 skill）

`skills/` 里存放的是面向 AI 执行的 SKILL.md——不是给人读的文章，而是可重复执行的工作流手册（例如"判断昇腾 NPU 型号与架构能力"）。`automatic-skill-routing/` 是配套的自动路由系统：它把所有来源的 skill 扫描成一份轻量目录（catalog），AI 拿到目录后按"召回 → 选择 → 知会 → 确认 → 加载 → 执行"的协议，根据你的自然语言任务自动判断该用哪些 skill。

- **怎么用**：把本仓库的 GitHub 链接告诉 AI 助手即可；AI 从 `llms.txt` 定位 skills 与路由说明。维护者也可以运行 `python3 automatic-skill-routing/skills-router/scripts/build_catalog.py` 生成目录，再把 `router-context.md` 注入给 AI。
- **能做什么**：你不需要记住 skill 的名字，用一句话描述任务，AI 就能自动匹配并加载对应工作流；新增一个 skill 也不需要修改任何路由代码。

#### 可选扩展：远程技能检索（默认关闭的预留扩展点）

自动技能路由还预留了一个运行期扩展点：`automatic-skill-routing/skills-router/runtime/`（可执行入口 `python3 -m runtime`）。它的设想是：当本地 skill 不够用时，在征得你同意后联网检索远程技能知识图谱（Ascend KG），把远程候选与本地目录合并后再路由。

诚实地说，它今天不是开箱即用的功能——远程服务端（ascend.wiki）从未实测过（仓库文档明确注明），仓库也不随附任何密钥。它的价值在于把同意门、隐私边界、降级回退这些容易写错的部分提前实现并测试好，等真实服务可用时即可激活。接口设计仅参照上游 [agent0/kg-tools](https://gitcode.com/agent0/kg-tools)——未集成其任何代码，也不调用 kg-tools 官方服务，整个运行时全部自研。

**若将来要启用，维护者需要两步配置**（环境前置：Python 3.10+，已安装 `pyyaml`）：

1. 生成本地目录（远程候选会和它合并）：

   ```bash
   python3 automatic-skill-routing/skills-router/scripts/build_catalog.py
   ```

2. 设置 Ascend KG 服务密钥（需先向 ascend.wiki 服务方申请）：

   ```bash
   export ASCEND_KG_API_KEY=<你的密钥>
   ```

   不设置或留空即为纯本地模式：完全不联网，也不影响本地路由。

**配置好后，由宿主 Agent 按任务调用**（一进程一任务的 NDJSON 会话，通常由 Agent 平台适配器自动驱动，不需要手动输入）：

```bash
cd automatic-skill-routing/skills-router
python3 -m runtime \
  --catalog catalog.json \
  --workspace-root ../.. \
  --native-skill <已安装的本地skill名>   # 可重复，声明宿主原生 skill 工具里已装的 skill
```

**任务执行过程中，你会被询问两次**，两次都拒绝也可以继续：

1. **网络同意**：先展示将精确发送到 `ascend.wiki` 的请求内容（只有当前任务文本，不含对话历史和文件内容），你同意后才发起检索；
2. **激活确认**：选中远程 skill 后、加载其内容前，再单独确认一次。

- **不启用会怎样**：完全不联网，自动路由照常基于本地目录工作，功能二不受任何影响。
- **启用后能做什么**：把 skill 的查找范围从本地目录扩展到远程知识图谱；密钥缺失、你拒绝同意、网络或服务出错时，自动回退为纯本地模式并向你说明原因，任务不会中断。远程内容只临时内联展示，不会安装到本地、不会落盘。
- **想先试试**：不设密钥跑 `python3 -m runtime --help` 可离线自检；纯本地完整流程见 `automatic-skill-routing/skills-router/opencode-adapter.md`。

## 目标

- 为系统方向新生提供一条从 AI 计算负载到推理、训练、Kernel、硬件、集群和 Benchmark 的学习路径。
- 为研究者整理系统方法、性能方法、实验方法、容量模型和故障复盘。
- 为 AI 检索、问答、知识图谱和 skills 提供可引用、可追溯的结构化知识源。

## 写作原则

- 每篇文档必须包含 front matter 元数据。
- 重要结论需要标明论文、代码仓库、硬件规格、Benchmark 或实验记录来源。
- 性能结论必须保留 workload、batch shape、sequence length、precision、硬件环境、软件版本、并发模型和复现状态。
- 技术比较需要说明前提假设、瓶颈类型、适用范围和反例。
- 临时讨论不要直接进入知识库，先沉淀成可复用的知识点、系统论文笔记、实验记录或技术决策。

## 推荐阅读路径

1. 从 `知识地图` 理解整体结构。
2. 新生先阅读 `入门导读`、`AI 基础概念`、`Transformer 流程与原理`、`训练过程与原理`、`推理过程与原理`、`多模态原理`。
3. 然后阅读 `AI 计算工作负载基础`，先用浅显方式理解模型、Transformer、训练、推理和多模态为什么成立。
4. 做推理方向重点阅读 `硬件基础（GPU/NPU）`、`推理系统与优化`、`Kernel、算子与编译优化`、`性能分析、Benchmark 与容量建模`。
5. 做训练和集群方向重点阅读 `训练系统与优化`、`集群、网络、存储与调度`、`可靠性、可观测性与故障复盘`。
6. 做硬件或算子方向重点阅读 `硬件基础（GPU/NPU）`、`Kernel、算子与编译优化`、`AI 加速器与计算架构`、`性能分析、Benchmark 与容量建模`。
7. 做论文复现和长期沉淀时，优先使用 `论文复现与系统案例`、`知识组织、模板与 AI 可读索引` 和模板。
