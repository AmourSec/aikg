# AI Knowledge Graph 续作指南

这份文档用于让任意 AI 工具或后来维护者接着建设本知识库。接手时先读本文件，再读 `README.md`、`mkdocs.yml`、`docs/knowledge-map.md`、`scripts/generate_llms_files.py` 和 `llms.txt`。

## 项目定位

本仓库是一个公开的 AI infrastructure 与高效计算知识库，面向系统方向研究生、工程师和 AI 辅助检索使用。内容重点不是训练出更“聪明”的模型，而是理解 AI 计算工作负载、硬件、推理系统、训练系统、Kernel、编译、集群、Benchmark、可靠性和知识沉淀方法。

写作风格应保持学术、技术、清晰，不写商业化宣传，不引入组织或公司背景。基础章节要让新手能建立直觉，深入章节要能支撑系统设计、性能分析和工程决策。

## 未来 AI 接手时的推荐提示词

```text
请先阅读 AI_CONTINUATION_GUIDE.md、README.md、mkdocs.yml、docs/knowledge-map.md、scripts/generate_llms_files.py 和 llms.txt。
这个仓库是公开 AI Infra / efficient computing 知识库，写作面向系统方向新生、工程师和 AI 代理。不要写商业宣传，不要提组织背景。
修改内容时保持 docs/ 文件路径、mkdocs 导航、知识地图、llms.txt / llms-full.txt 的一致性。新增文章优先放到所属章节目录，导航归属要和文件路径一致。完成后运行生成索引、MkDocs strict build 和 git diff 检查。
```

## 目录与职责

- `docs/`：文档站源文件，是给人阅读、也给 AI 检索的主要 Markdown 内容。
- `mkdocs.yml`：MkDocs Material 站点配置和导航结构，站点地址是 `https://amoursec.github.io/aikg/`。
- `docs/knowledge-map.md`：知识地图和导航型总览。章节结构、学习路径或重点主题变化时要同步更新。
- `llms.txt`：给 AI 的入口索引，列出重点文档、skills 和原始 Markdown 地址。
- `llms-full.txt`：给 AI 的单文件聚合上下文。
- `scripts/generate_llms_files.py`：生成 `llms.txt` 和 `llms-full.txt` 的脚本。新增重要主题时要维护 `PRIORITY_DOCS`、`PRIORITY_SKILLS` 和 `DESCRIPTIONS`。
- `skills/`：面向 AI 工具执行具体任务的 skill 样例和后续沉淀位置。
- `docs/99-templates/`：知识点、ADR、Benchmark 报告等模板。
- `site/`：MkDocs 构建产物，不是内容源头。

## 当前信息架构原则

导航结构按读者理解顺序组织：

1. `入门导读`：告诉新读者怎么用知识库。
2. `AI 计算工作负载基础`：只讲 AI、Transformer、训练、推理、多模态的必要直觉，要求浅显易懂，不追求理论深度。
3. `硬件基础`：介绍 GPU、NPU、GPU/NPU 对比、Ascend/CANN 和硬件适配 skill 样例。虽然文件夹名是 `12-hardware-basics`，但导航上放在推理系统前面，保持学习顺序。
4. `推理系统与优化`：讲请求生命周期、Prefill/Decode、调度、KV Cache、PagedAttention、量化、Speculative Decoding、PD 分离、MoE、vLLM、TensorRT-LLM、SGLang、RAG/Agent、Benchmark 等。
5. `训练系统与优化`：讲训练任务生命周期、数据管线、batch、显存、并行策略、通信重叠、FLUX、Muon、checkpoint、benchmark、DeepSpeed/Megatron/FSDP 等。
6. `Kernel、算子与编译优化`：讲 Attention 计算模式、Triton、TorchInductor、MLIR、TileLang、MegaKernel、Persistent Kernel 和自动生成。
7. `AI 加速器与计算架构`、`集群、网络、存储与调度`、`性能分析、Benchmark 与容量建模`、`可靠性、可观测性与故障复盘`：支撑更大系统的架构和运营视角。
8. `论文复现与系统案例`、`知识组织、模板与 AI 可读索引`、`模板`：支撑长期知识沉淀、决策复盘和 AI 可读化。

维护规则：

- `mkdocs.yml` 中的导航归属应与文件路径归属一致。例如 `docs/11-knowledge-index/skills-authoring-guide.md` 应放在 `知识组织、模板与 AI 可读索引` 下，而不是跨挂到 `入门导读`。
- 如果某篇文章只是在学习路径中推荐，可以在正文或知识地图中链接，不要用跨章节导航制造路径和归属不一致。
- 稳定 URL 和稳定文件路径优先于纯粹编号美观。不要轻易重命名章节目录。

## 内容写作约定

- 面向新手的科普内容要解释“流程是什么、每一步为什么成立、数据从哪里来、模型内部是什么数字表示”，避免一上来堆专有名词。
- 深入系统章节要讲清楚 workload、数据流、控制流、状态、瓶颈、指标、常见优化方向、适用边界和排查方法。
- 硬件文章优先引用官方手册、官方架构图或权威资料；如果使用外部图，应标注来源和用途。对关键硬件单元要解释用途、常见操作和与 AI workload 的关系。
- 涉及软件版本、硬件型号、性能特性、工具能力时，应查官方或最新可信来源，不凭记忆写。
- 图示优先用 Mermaid 或可维护的本地资源。图要服务理解，不要只做装饰。
- 不要把所有内容写成“百科词条”。本项目更关注 AI 计算系统如何运行、为什么慢、如何测、如何优化。

## AI Skills 约定

不是每篇文章都需要 skill。科普文章主要给人和 AI 检索理解，不要强行配 skill。

适合写成 skill 的内容通常满足以下条件：

- 它是一个可重复执行的工程流程，而不是单纯概念解释。
- 输入、判断步骤、输出格式比较稳定。
- 希望 AI 在后续工作中按同一方法检查、适配、诊断或生成结果。

当前样例：

- 指南文章：`docs/11-knowledge-index/skills-authoring-guide.md`
- 硬件适配说明：`docs/12-hardware-basics/ai-skills-sample.md`
- Skill 样例：`skills/npu-arch-capability-check/SKILL.md`

新增 skill 的基本结构：

```text
skills/<skill-name>/
  SKILL.md
  references/       # 可选，放检查表、表格、示例材料
  scripts/          # 可选，放可复用脚本
  assets/           # 可选，放小型辅助资源
```

`SKILL.md` 的 frontmatter 只保留必要字段：

```yaml
---
name: skill-name
description: 一句话说明这个 skill 何时应被使用
---
```

正文应写清楚触发条件、输入材料、执行步骤、输出格式、边界条件和验证方式。Skill 不是文章摘要，而是让 AI 能执行任务的操作手册。

## 新增或修改文章的流程

1. 先读所属章节的 `index.md`、相邻文章、`mkdocs.yml` 和 `docs/knowledge-map.md`，确认这篇内容应该放在哪里。
2. 在 `docs/<chapter>/` 下新增或修改 Markdown 文件。
3. 在 `mkdocs.yml` 中把文章加入对应章节，保持导航归属和文件路径一致。
4. 如果影响章节结构、学习路径或重点主题，同步修改 `docs/knowledge-map.md`。
5. 如果是重要主题，同步修改 `scripts/generate_llms_files.py` 中的 `PRIORITY_DOCS`、`PRIORITY_SKILLS` 或 `DESCRIPTIONS`。
6. 重新生成 AI 索引文件。
7. 运行严格构建和格式检查。
8. 提交前检查 git diff，确认没有误改生成物、无关文件或用户未要求处理的补丁。

常用命令：

```bash
python3 scripts/generate_llms_files.py
.venv/bin/mkdocs build --strict
git diff --check
git status --short --branch
```

## 本地预览与部署

本地预览端口固定为 `8801`：

```bash
.venv/bin/mkdocs serve
```

打开：

```text
http://127.0.0.1:8801/
```

源码仓库：

```text
git@github.com:AmourSec/aikg.git
```

公开站点地址：

```text
https://amoursec.github.io/aikg/
```

站点由本仓库的 `.github/workflows/deploy-pages.yml` 自动部署：推送到 `master` 即构建并发布到 GitHub Pages 项目地址（`/aikg` 子路径），每日定时任务还会刷新文章浏览量快照。

旧的根域名 `https://amoursec.github.io/` 由 `AmourSec/AmourSec.github.io` 仓库托管，其内容已替换为跳转页（`index.html` 整页跳转、`404.html` 按原始路径跳转到 `/aikg` 下对应页面），不再承载站点内容。不要再用 rsync 手动同步站点到 Pages 仓库；只改根目录维护文档或 README 时，通常只需要推送源码仓库。

## 质量检查清单

每次完成变更前至少检查：

- `mkdocs build --strict` 是否通过。
- `git diff --check` 是否无空白错误。
- 新增文件是否被 `mkdocs.yml`、知识地图或 `llms.txt` 正确引用。
- 导航标题、文件名、页面标题是否一致或至少不让维护者困惑。
- 没有把某篇文章挂到与文件路径不一致的章节下。
- 没有把临时文件、实验输出、无关补丁加入提交。

可用下面的思路检查导航目标是否存在：用 Python 读取 `mkdocs.yml`，递归展开 `nav`，确认每个 Markdown 路径都存在于 `docs/` 下。由于 `mkdocs.yml` 里有 `!!python/name:mermaid2.fence_mermaid` 自定义 tag，读取时应使用 `yaml.BaseLoader`。

## Git 注意事项

- 当前主分支是 `master`。
- 不要擅自重置、回滚或删除用户已有改动。
- 若看到未跟踪的 `sanitize-for-intranet.patch`，除非用户明确要求，否则不要添加、修改或删除它。
- 手工改文件时优先使用补丁方式，保持 diff 清晰。
- 提交要小而聚焦，提交信息说明实际变更。

## 维护心法

这个知识库要同时服务两类读者：人需要清晰的学习路径，AI 需要稳定、结构化、可引用的上下文。因此每次新增内容都要同时考虑：

- 人能不能从导航和知识地图知道它在哪里、为什么要读。
- AI 能不能从 `llms.txt`、标题、描述和原始 Markdown 路径准确找到它。
- 后续维护者能不能只看文件路径和导航就知道应该在哪里继续补充。

优先保持结构干净、路径稳定、解释清楚。不要为了短期方便制造重复入口、跨章节挂载或难以追踪的重定向。
