---
title: 自动 Skill 发现、选择与按需加载系统设计
status: approved-design
owner: maintainers
created: 2026-08-03
updated: 2026-08-03
---

# 自动 Skill 发现、选择与按需加载系统设计

## 1. 摘要

本设计为 AI Knowledge Graph 增加一层与模型、Agent 应用和 API 平台无关的 Skill 路由能力。用户只需描述任务，不需要预先知道有哪些 Skill。系统负责发现 Skill、检索候选、验证适用性、选择最小必要集合、向用户披露即将使用的 Skill 名称，然后按需加载完整 `SKILL.md` 及其明确引用的资源。完成交接后，具体追问、执行和验证由大模型与已选择的 Skill 共同处理。

系统采用三级渐进式披露：

1. 用轻量目录召回候选；
2. 读取少量候选的完整 `SKILL.md`，验证适用范围；
3. 向用户知会或请求确认后，加载最终 Skill 的必要资源并交给大模型。

本设计把可扩展性和可维护性作为硬性约束：新增一个标准 Skill 不修改路由代码；新增一个 Skills 仓库只修改来源配置；文档、Schema、示例和测试共同定义维护契约；任何失败都不能发布半成品目录。

## 2. 背景与现状

当前仓库已经建立了清晰分层：

- `docs/` 保存供人和 AI 检索的知识；
- `skills/` 保存可重复执行的 AI 工作流；
- `llms.txt` 和 `llms-full.txt` 提供 AI 可读入口；
- `scripts/generate_llms_files.py` 生成 AI 索引；
- `AI_CONTINUATION_GUIDE.md` 约束后续人工或 AI 维护行为。

仓库已有 `skills/npu-arch-capability-check/SKILL.md` 样例，但 Skill 入口仍通过 `PRIORITY_SKILLS` 静态列举。这个方式适合少量本地 Skill，不适合接入 `Ascend/agent-skills`、`cann/cannbot-skills` 以及未来持续增长的第三方来源。系统需要从“静态列举 Skill”演进为“动态发现、标准化建库、检索选择、按需加载”。

## 3. 目标

系统必须满足以下目标：

1. 用户不需要知道 Skill 名称，只需描述任务。
2. 系统自动寻找、选择并使用合适的 Skill。
3. 在实际启用具体 Skill 前，必须向用户披露 Skill 名称和用途；被标记为需要确认的 Skill 必须等待用户同意。
4. 新增标准 `SKILL.md` 后，无需修改路由代码即可参与检索。
5. 新增外部 Skills 仓库时，只需增加来源配置。
6. 支持选择单个 Skill，也支持按明确顺序组合最小必要 Skill 集合。
7. 支持关键词检索与语义检索；语义检索不可用时仍能降级工作。
8. 对重复、同名、损坏、断链和版本漂移给出确定、可审计的处理结果。
9. 构建、校验、评测和发布必须可重复、可回滚、可由新人或 AI 严格执行。
10. 核心目录、选择与交接协议不绑定特定模型、Embedding 服务、Agent App 或工具调用框架。

## 4. 非目标

本阶段不负责：

- 设计 Skill 内部的具体业务流程；
- 代替 Skill 决定应该运行哪些命令或修改哪些文件；
- 实现通用 Shell、文件、网络、NPU 或代码执行沙箱；
- 统一所有第三方 Skill 的写作风格；
- 把几百个完整 Skill 永久塞入每次模型上下文；
- 要求用户从 Skill 列表中人工挑选；
- 建设复杂的领域知识图谱或通用工作流编排语言。

路由系统只负责发现、选择、披露、加载和交接。交接之后的执行行为由大模型、Skill、本地策略和实际运行环境共同决定。

## 5. 术语

| 术语 | 定义 |
| --- | --- |
| Source | 一个本地目录或固定版本的外部 Git 仓库，可能包含多个 Skill。 |
| Skill | 以一个 `SKILL.md` 为入口的任务工作流目录。 |
| Catalog Entry | 路由系统为一个 Skill 生成的规范化轻量记录。 |
| Override | 不修改上游仓库、只在本仓补充或纠正 Skill 元数据的受控配置。 |
| Source Lock | 记录每个来源实际解析到的不可变版本。 |
| Candidate | 轻量检索召回、尚未最终启用的 Skill。 |
| Selected Skill | 已通过完整 `SKILL.md` 适用性验证的 Skill。 |
| Activation | 将已选 Skill 的指令交给大模型并开始遵循其工作流。 |
| Disclosure Gate | Activation 前向用户披露 Skill 名称并按策略知会或请求确认。 |
| Handoff | 路由器把原始任务、已选 Skill 和加载结果交给大模型。 |
| Quarantine | Skill 存在格式、引用或完整性错误，暂不允许参与路由的状态。 |

候选验证阶段读取 `SKILL.md` 仅用于判断是否适用，不算 Activation。只有通过 Disclosure Gate 并把指令加入任务执行上下文后，才视为开始使用具体 Skill。

## 6. 不变量

实现和维护不得破坏以下不变量：

1. 路由代码中不得出现具体业务 Skill 名称的判断分支。
2. 生成目录和缓存不得手工修改。
3. 相同来源版本与配置必须生成字节一致的规范化目录。
4. 未通过完整适用性验证的候选不得进入 Activation。
5. 未完成用户披露的 Skill 不得进入 Activation。
6. `activation_policy: confirm` 的 Skill 未获得同意不得进入 Activation。
7. 部分构建结果不得替换上一版可用目录。
8. 路由器不得虚构不存在的 Skill、路径、依赖或来源版本。
9. Skill 引用只允许在其受信任来源根目录内解析，不允许路径越界。
10. 用户拒绝某个 Skill 后，本次任务不得再次选择同一 Skill，除非用户明确撤销拒绝。
11. 新增 Skill 或来源不得要求修改选择算法。

## 7. 方案比较与决策

### 7.1 全量提示词

把所有 Skill 名称、描述或正文直接写入系统提示词。实现简单，但 Skill 数量增长后会显著占用上下文，更新成本高，候选之间容易互相干扰，无法满足长期扩展要求。

### 7.2 两阶段检索与按需加载

先检索轻量目录，再读取少量候选正文完成适用性验证，最后加载被选中的资源。该方案支持渐进式披露、增量索引、无语义服务降级和来源扩展，是本设计采用的方案。

### 7.3 Skill 知识图谱

显式建模任务、能力、前置条件、冲突和组合关系。它适合未来处理非常复杂的跨领域编排，但初期需要大量人工维护，且第三方 Skill 元数据不完整，因此不在首期范围内。

## 8. 总体架构

系统由五个边界清晰的组件组成。

### 8.1 Skill Source Registry

Source Registry 读取 `skill-system/sources.yaml`，提供所有来源的声明配置。它只负责描述来源，不负责理解任何具体 Skill。

来源至少包含：

- 稳定 `source_id`；
- `local` 或 `git` 类型；
- 本地路径或仓库 URL；
- 期望分支、标签或提交；
- include/exclude 路径规则；
- 是否启用；
- 来源优先级；
- 可选 canonical 来源关系。

Git 来源还可以声明是否递归初始化 submodule。对于 `Ascend/agent-skills` 这类聚合仓库，Source Registry 必须同时锁定父仓提交与每个实际初始化的 submodule 路径、URL 和提交；只锁父仓分支名或只克隆父仓而不物化 submodule 都不算完成同步。

解析后的不可变提交先写入 staging 中的候选 lock。构建和运行使用 lock 中的版本，而不是每次隐式追踪浮动分支。仓库根部的 `sources.lock.json` 是 active 版本的公开镜像，只有完整构建通过并切换 active 后才更新；同步或构建失败不得提前覆盖它。

### 8.2 Catalog Builder

Catalog Builder 在每个启用来源中递归发现严格命名的 `SKILL.md`，完成：

- Front Matter 解析；
- 名称、描述和 Scope 提取；
- 稳定 `skill_id` 生成；
- 内容哈希与引用清单生成；
- Override 合并；
- 重复和同名检测；
- 引用边界与完整性检查；
- Catalog Entry 输出；
- 隔离清单与构建报告输出。

Builder 不根据领域或目录名硬编码业务分类。

### 8.3 Retrieval Index

Retrieval Index 为规范化 Catalog Entry 建立两类索引：

- 关键词索引：覆盖名称、描述、Scope 摘要、别名、标签和关键技术词；
- 语义索引：覆盖描述与适用范围的语义表示。

关键词与语义后端都通过稳定接口接入。语义后端不可用时，系统必须回退到关键词索引，而不是停止整个 Skill 路由能力。

### 8.4 Skill Router

Skill Router 接收用户原始任务，完成候选召回、完整适用性验证、最小集合选择、顺序确定和结构化决策输出。Router 不执行 Skill 工作流。

### 8.5 Lazy Loader

Lazy Loader 提供两个严格分离的只读模式：

- `inspect`：候选验证时只读取候选的 `SKILL.md`，不加载引用资源、不执行工作流，也不产生 Handoff；
- `activate`：通过 Disclosure Gate 后，校验已选 Skill，加载允许的显式引用并生成 Handoff Bundle。

Loader 负责路径安全、循环检测、大小控制、内容与资源哈希校验，不运行引用中的脚本。`inspect` 的读取不算 Activation，`activate` 才会把指令交给大模型。

“显式引用”只包括能够确定解析的来源内相对路径：Markdown 链接或图片目标、标准资源清单中的路径，以及反引号包裹且实际存在的相对文件路径。Loader 不根据自然语言猜测文件名，也不自动加载整个目录；需要整目录时，Skill 必须明确声明目录或具体文件清单。

## 9. 建议目录结构

```text
skill-system/
  README.md
  sources.yaml
  sources.lock.json
  router-instructions.md
  schemas/
    sources.schema.json
    source-lock.schema.json
    override.schema.json
    quarantine-ack.schema.json
    catalog-entry.schema.json
    routing-case.schema.json
  overrides/
    <source-id>/
      quarantine-ack.yaml
  examples/
    sources.valid.yaml
    sources.invalid.yaml
    overrides.valid.yaml
    sample-catalog-entry.json
  generated/
    catalog.jsonl
    catalog-summary.md
    quarantine.json
    build-report.json
    index/
  cache/
    sources/
    staging/
  evals/
    routing-cases.yaml
    regression-baseline.json
docs/11-knowledge-index/
  skill-routing-system.md
  skill-maintenance-runbook.md
```

`cache/` 是可删除的同步缓存，`generated/` 是确定性构建产物。两者都不得成为手工维护的事实来源。

## 10. 数据契约

### 10.1 Source 配置

规范化 Source 配置示例：

```yaml
schema_version: 1
sources:
  - source_id: local-aikg
    type: local
    location: skills
    enabled: true
    priority: 100
  - source_id: ascend-agent-skills
    type: git
    location: https://gitcode.com/Ascend/agent-skills.git
    revision: master
    submodules: recursive
    enabled: true
    priority: 50
    exclude:
      - official/CANNBot/**
  - source_id: cannbot
    type: git
    location: https://gitcode.com/cann/cannbot-skills.git
    revision: master
    enabled: true
    priority: 80
```

同时登记两个仓库时，从聚合仓库排除 `official/CANNBot/**`，以直接 CANNBot 仓库为 canonical 来源。这个关系存在于配置中，不进入路由代码。如果只登记 `agent-skills`，则可以不排除其 CANNBot 子目录。聚合仓库通过 submodule 提供内容时，lock 必须记录父仓与实际 submodule 提交，Catalog Entry 同时保留父仓 revision 以及所属 submodule 路径和 revision。

凭证不得写入来源文件。私有来源的鉴权由外部环境提供。

### 10.2 Catalog Entry

每个有效 Skill 生成一个 Catalog Entry：

```json
{
  "schema_version": 1,
  "skill_id": "cannbot:ops/ascendc-env-check",
  "name": "ascendc-env-check",
  "description": "...",
  "scope_summary": "...",
  "source_id": "cannbot",
  "source_revision": "immutable-commit",
  "submodule_path": null,
  "submodule_revision": null,
  "relative_path": "ops/ascendc-env-check/SKILL.md",
  "content_hash": "sha256:...",
  "bundle_hash": "sha256:...",
  "status": "enabled",
  "activation_policy": "notify",
  "repeatable": false,
  "aliases": [],
  "tags": [],
  "dependencies": [],
  "conflicts": [],
  "references": [],
  "canonical_skill_id": "cannbot:ops/ascendc-env-check"
}
```

`skill_id` 使用 `source_id` 与相对目录组合，保证同名 Skill 可共存。`name` 保留上游名称，用于向用户披露。`source_revision` 是父来源不可变提交；Skill 位于 submodule 时同时记录 `submodule_path` 与 `submodule_revision`。`content_hash` 只覆盖规范化后的 `SKILL.md`；`bundle_hash` 覆盖 `SKILL.md` 与全部已解析显式引用相对于 Skill bundle 的路径、内容和顺序，用于完整加载校验与跨聚合路径的 exact duplicate 判断。用于实际加载的 source-relative 路径另行保留，不能混入 bundle-relative 去重哈希。

### 10.3 Override

Override 只用于：

- 补充别名、标签和跨语言触发词；
- 显式声明依赖、冲突或 canonical 关系；
- 禁用已知损坏或不适用 Skill；
- 设置 `activation_policy`；
- 在有明确证据时声明同一路由链中是否允许重复调用，默认不允许；
- 在不修改上游的情况下修正可验证的元数据错误。

自动生成的模型推断不得直接成为生效 Override。任何会改变选择、依赖、冲突、启用状态或确认策略的 Override 必须作为版本化文件提交。

### 10.4 Activation Policy

Activation Policy 有且只有两种：

- `notify`：向用户知会名称和用途后立即启用；这是默认值。
- `confirm`：向用户披露名称和用途，等待明确同意后启用。

本系统不根据 Skill 名称猜测确认策略。策略来自标准元数据或受版本控制的 Override。

## 11. 构建与更新流程

一次目录更新必须按以下顺序进行：

1. 校验 `sources.yaml` 和 Override Schema；
2. 将所有来源同步到独立 staging 目录；
3. 把浮动 revision 以及启用的 submodule 解析为不可变提交，生成 staging 候选 lock；
4. 递归发现 `SKILL.md`；
5. 解析、标准化并校验每个 Skill；
6. 验证显式引用、路径边界和循环；
7. 合并已审核 Override；
8. 处理 exact duplicate、canonical 关系和同名冲突；
9. 生成候选 Catalog、lock、隔离清单和构建报告；
10. 构建关键词与语义索引；
11. 运行契约测试和路由回归评测；
12. 检查生成差异与退化报告；
13. 所有目录发布门禁通过后原子发布包含 lock、Catalog 和索引的完整版本；
14. 保存上一版可回滚快照。

任一步失败都不得修改当前 active lock、Catalog 和索引。单独执行 source sync 只生成候选 lock，不改变根部 active lock 镜像。

## 12. 路由运行流程

### 12.1 第一级：轻量召回

Router 保留用户原始任务，分别进行关键词与语义召回。建议默认各召回 12 条，合并去重后进入下一阶段。数量是配置项，不影响协议。

召回前必须过滤：

- disabled；
- quarantined；
- 被 canonical Entry 替代的副本；
- 用户在本次任务中已拒绝的 Skill；
- 已在当前路由链中完成且不应重复调用的 Skill。

### 12.2 第二级：完整适用性验证

Router 对排序靠前的少量候选调用 Loader 的 `inspect` 模式，只读取完整 `SKILL.md` 进行判断。建议默认最多检查 5 个候选。此时不加载引用资源、不通过 Disclosure Gate、不产生 Handoff。Selector 输出结构化决策：

```json
{
  "decision": "selected",
  "skills": [
    {
      "skill_id": "cannbot:ops/ascendc-env-check",
      "role": "primary",
      "order": 1,
      "reason": "任务要求验证 Ascend C 环境，符合该 Skill 的 Scope"
    }
  ],
  "confidence": "high",
  "clarification": null
}
```

`decision` 允许：

- `selected`：已有足够证据选择 Skill；
- `no_match`：没有适用 Skill，交回普通模型能力；
- `clarification_required`：任务边界不足，只能询问任务细节。

澄清问题不得要求用户了解或挑选 Skill。

### 12.3 选择规则

Selector 必须遵守：

1. 不得只根据名称相似选择。
2. 必须检查 Scope、触发条件和排除条件。
3. 具体 Skill 优先于宽泛 Skill。
4. 一个 Skill 足以覆盖任务时不得增加冗余 Skill。
5. 只有不同步骤或明确依赖才组合多个 Skill。
6. 组合顺序必须解释为 prerequisite、primary 或 supporting。
7. 来源优先级只能用于匹配质量相当时的稳定决胜，不能覆盖适用性。
8. Skill 描述不足以确认时，必须读取正文；仍不足则返回 `no_match` 或询问任务细节。
9. 不得推断不存在的依赖、冲突或引用。

### 12.4 Disclosure Gate

选择完成后、Activation 之前，系统必须向用户披露：

- Skill 的公开 `name`；
- 每个 Skill 在当前任务中的一句话用途；
- 多 Skill 的使用顺序；
- 是否需要确认。

默认知会文案：

```text
我将使用以下 Skill 完成任务：
- ascendc-env-check：检查 NPU 设备与 CANN/Ascend C 环境。
现在开始执行该 Skill 的工作流。
```

确认文案：

```text
计划使用以下 Skill：
- <skill-name>：<当前任务中的用途>。
该 Skill 配置为使用前确认。是否继续？
```

多 Skill 时按 `order` 列出。用户只负责知情或批准，不需要从目录中选择。

Disclosure 记录必须绑定当前 session、任务、按顺序排列的 Skill ID、选择哈希和实际展示文案哈希，不能跨任务复用。`notify` 只有在调用方已把原文展示给用户后才能记为 `notified`；`confirm` 的 `confirmed` 或 `rejected` 必须来自调用方捕获的明确用户决定。核心协议不假装能够观察某个 UI，但 Loader 必须拒绝缺失、错序、哈希不符或跨 session 的记录。

若用户拒绝：

1. 将被拒绝 Skill 加入本次任务排除集；
2. 重新路由一次，寻找真正适用的替代 Skill；
3. 没有替代项时，以无 Skill 模式继续或说明无法按 Skill 工作流完成；
4. 不得暗中重新启用被拒绝 Skill。

二次路由发现新的辅助 Skill 时，也必须在 Activation 前单独通过 Disclosure Gate。

### 12.5 第三级：按需加载与 Handoff

Disclosure Gate 通过后，Lazy Loader 才进入 `activate` 模式：

1. 校验已选 Skill 的来源版本、`content_hash` 和 `bundle_hash`；
2. 加载完整 `SKILL.md`；
3. 只跟随正文明确引用的资源；
4. 检测断链、循环、越界和内容上限；
5. 生成 Handoff Bundle；
6. 将任务控制权交给大模型与 Skill。

Handoff Bundle 至少包含：

```json
{
  "task": "用户原始任务",
  "decision": "selected",
  "skills": [
    {
      "skill_id": "cannbot:ops/ascendc-env-check",
      "name": "ascendc-env-check",
      "source_revision": "immutable-commit",
      "role": "primary",
      "order": 1
    }
  ],
  "disclosure": {
    "policy": "notify",
    "status": "notified"
  },
  "handoff_status": "ready"
}
```

上例只展示控制字段。实际 Handoff 还必须包含每个已选 Skill 经哈希校验的完整 `SKILL.md` 文本，以及已加载显式资源的稳定路径、内容哈希、媒体类型、编码和内容，保证接手模型不依赖未声明的本地文件读取。任一资源失败时不得交付部分 Bundle。

确认型 Skill 的 `disclosure.status` 必须为 `confirmed`。Loader 不得接受 `pending`、`rejected` 或缺失状态。

### 12.6 子任务再路由

Skill 执行中出现新的独立子任务时，大模型可以再次调用 Router。再路由输入必须包含：

- 原始任务；
- 当前子任务；
- 已选择和已完成 Skill；
- 被拒绝 Skill；
- 当前执行状态；
- 当前路由深度。

系统必须设置可配置的最大路由深度和已访问集合。达到上限、重复状态或检测到循环时，停止再路由并返回当前执行上下文，不进行隐式无限递归。

## 13. 去重与来源关系

### 13.1 完全重复

规范化内容与资源哈希完全一致时，折叠为一个 canonical Entry，同时记录全部来源。优先级更高或显式声明为 canonical 的来源负责实际加载。

### 13.2 聚合仓库重复

`Ascend/agent-skills` 通过 `official/CANNBot` 引用 CANNBot。两个来源同时启用时，通过 Source 的 exclude 规则排除聚合仓中的重复子树，使用直接 CANNBot 仓库作为 canonical 来源。这个规则记录在 `sources.yaml` 示例与维护 Runbook 中。

### 13.3 同名不同内容

同名但内容不同的 Skill 保留为不同 Entry。Router 依据 Scope、来源和任务匹配选择；内部日志使用完整 `skill_id`，用户披露使用 `name`。如果同一轮披露中出现同名 Skill，必须附加来源简称避免歧义。

## 14. 可扩展性设计

### 14.1 新增本地 Skill

把含有效 `SKILL.md` 的目录放入已登记的本地 Source，重新构建即可。不得增加路由条件分支。

### 14.2 新增外部仓库

只在 `sources.yaml` 增加 Source，生成 lock，并运行全套门禁。解析、检索、选择和加载协议不变。

### 14.3 新增元数据能力

Catalog Entry 通过 `schema_version` 演进。新增可选字段必须保持旧记录可读取；新增必填字段必须提供迁移器和版本升级文档。

### 14.4 更换检索实现

关键词、Embedding、重排和 Selector 通过接口替换。Catalog、Disclosure 和 Handoff 数据契约保持稳定。

### 14.5 跨语言扩展

上游描述保留原文。中文别名、术语和触发表达可通过受审核 Override 补充。自动翻译或模型生成标签只能作为建议，不能未经审核改变生效目录。

## 15. 异常处理

### 15.1 Source 同步失败

首次同步失败时不创建该来源。更新同步失败时保留上一版 lock 和 active Catalog，报告 stale 状态并使本次发布失败。

### 15.2 无效 Skill

以下情况进入 Quarantine：

- 缺失或无法解析的 Front Matter；
- 缺少有效 `name` 或 `description`；
- 路径与名称违反基本契约；
- 显式引用不存在；
- 引用循环或越界；
- 内容无法按声明编码读取。

默认 strict 模式下，新增校验错误阻止发布。批量首次接入外部来源时，可以通过 `overrides/<source-id>/quarantine-ack.yaml` 明确接受已知隔离项；该文件必须列出稳定 Skill 路径、预期错误码、来源 revision 范围、理由和审核日期，并通过 `quarantine-ack.schema.json`。任何未被精确列出的新隔离仍阻止发布。不得用忽略全部错误的全局开关绕过门禁。

### 15.3 索引失败

任一必需索引构建失败时不发布。语义后端明确配置为 optional 且关键词索引通过时，可以发布降级版本，并在报告中标记 degraded。

### 15.4 候选不明确

Selector 返回 `clarification_required`，只询问任务事实。用户回答后重新路由。不得展示目录要求用户挑选。

### 15.5 Loader 失败

Loader 不得交付残缺 Skill。失败后可以重新评估仍通过验证的独立候选；如果没有可靠替代项，返回无 Skill 或明确失败，不得拼接部分指令继续。

### 15.6 用户拒绝

拒绝只影响当前任务路由状态，不修改全局 Catalog。拒绝事件写入任务日志，用于解释为什么选择替代路径。

## 16. 文档与 AI 维护契约

文档是系统的一部分，必须与实现共同通过测试。

### 16.1 文档职责

- 根 `README.md`：系统价值、最短入口和文档链接；
- `AI_CONTINUATION_GUIDE.md`：AI 接手前必读、禁止事项和完成门禁；
- `docs/11-knowledge-index/skill-routing-system.md`：面向人和 AI 的架构与运行协议；
- `docs/11-knowledge-index/skill-maintenance-runbook.md`：新增、更新、删除、禁用、回滚与排障；
- `skill-system/README.md`：精确命令、参数、预期输出和快速检查；
- `schemas/`：机器可执行的数据契约；
- `examples/`：同时服务文档和自动化测试的正反例。

### 16.2 Runbook 固定章节结构

每个操作场景必须包含：

```text
目的
前置条件
允许修改的文件
禁止修改的文件
精确操作步骤
成功时的预期输出
失败原因与处理
验证命令
回滚方法
提交前检查
```

Runbook 必须覆盖：

1. 新增本地 Skill；
2. 新增外部 Source；
3. 更新外部版本；
4. 禁用、恢复或删除 Skill；
5. 添加 Override；
6. 处理重名与重复；
7. 处理 Quarantine；
8. 重建目录与索引；
9. 添加路由评测；
10. 分析路由退化；
11. 回滚错误发布。

### 16.3 AI 强制维护协议

`AI_CONTINUATION_GUIDE.md` 必须明确：

```text
修改 Skill 系统前，先阅读系统设计和维护 Runbook。
不得手工编辑 generated/ 和 cache/。
不得通过硬编码具体 Skill 名称修复路由问题。
新增或更新 Skill 后必须重建目录并运行路由回归测试。
遇到来源、重复、引用或评测结果不明确时停止并报告，不得猜测。
只有所有门禁通过后才能宣称更新完成。
```

### 16.4 防止文档漂移

- 文档中的配置样例直接作为 Schema 测试数据；
- 文档引用的命令必须有对应 CLI 帮助或 smoke test；
- 生成文件包含“禁止手工修改”和生成器版本；
- Schema、命令或工作流改变时，相关文档必须在同一变更中更新；
- AI 索引必须包含 Skill Routing Protocol 和维护文档入口；
- 外部 Skill 正文不全部内联进每次上下文，Catalog 和逐文件来源必须保持可寻址。

## 17. 测试策略

### 17.1 单元测试

覆盖：

- Front Matter 解析；
- 稳定 ID；
- 内容与资源哈希；
- include/exclude；
- Override 合并；
- 路径边界、断链和循环；
- exact duplicate 与同名不同内容；
- lock 解析；
- 增量更新；
- 确定性输出；
- Disclosure 状态机；
- 再路由循环检测。

### 17.2 契约测试

- Source、lock、Override、Quarantine acknowledgement、Catalog 和 Routing Case 必须通过对应 JSON Schema；
- 正例必须通过，反例必须以预期错误码失败；
- 相同输入两次构建的输出必须字节一致；
- 所有 enabled Entry 必须可以完整加载；
- `confirm` Entry 的未确认 Handoff 必须被拒绝。

### 17.3 路由评测

每个 Routing Case 至少包含：

- 用户原始问题；
- 预期 Skill 或允许集合；
- 明确禁止选择的 Skill；
- 是否预期无匹配；
- 多 Skill 时的允许顺序；
- 选择理由所需的关键证据。

评测指标：

- Recall@K；
- Top-1 最终选择准确率；
- 无匹配误选率；
- 禁止 Skill 选择率；
- 多 Skill 顺序准确率；
- Disclosure 完成率；
- 完整加载成功率；
- 新 Skill 对既有样例的抢占回归。

### 17.4 扩展性测试

必须自动验证：

- 新增 Skill 后不修改 Router 也能发现；
- 新增 Source 后只改配置即可发现；
- 删除 Source 不影响其他来源；
- 语义后端不可用时关键词路径可工作；
- 更换检索后端不改变 Catalog 和 Handoff 契约；
- Skill 总量扩大后仍采用有限候选和渐进加载，不全量注入上下文。

## 18. 质量门槛

首期完整生产资格必须达到：

- enabled Skill 目录解析成功率 100%；
- Selected Skill 完整加载成功率 100%；
- Disclosure Gate 完成率 100%；
- Routing Gold Set Recall@10 不低于 95%；
- 最终 Top-1 选择准确率不低于 90%；
- 无匹配误选率不高于 5%；
- 明确禁止 Skill 的选择率为 0%；
- 所有显式引用可解析且不越界；
- 相同输入生成结果字节一致；
- 门禁失败时 active 版本保持不变。

门禁分为两个明确层级：

- `catalog` 发布门禁：校验来源、解析、引用、Disclosure/Loader 契约、确定性和 Recall@10。该层通过后可以发布可用的 active 目录与索引；
- `production` 模型资格门禁：在具体部署所用模型或 Selector adapter 的已捕获结果上，额外校验最终 Top-1、无匹配误选、禁止选择和多 Skill 顺序。

核心系统不绑定模型 API，因此没有部署模型结果时，模型相关指标必须记录为 `not_run`，不能伪装为通过。此时可以说明“目录已发布”，但不得说明“模型选择效果已通过生产资格”。一旦声明某个模型或 adapter 用于生产，就必须提供其评测结果并通过 `production` 门禁。

本地新增或实质修改 Skill 时，必须增加正向和负向路由样例。批量同步外部来源时，至少为发生变化且可能与现有能力重叠的 Skill 增加回归样例。

## 19. 发布与回滚

发布使用版本化 staging 和 active 指针或等价的原子替换机制：

1. 在 staging 中生成完整候选版本；
2. 运行全部测试和评测；
3. 生成差异、来源版本和指标报告；
4. `catalog` 门禁通过后切换 active；若要声明生产可用，还必须同时满足 `production` 模型资格门禁；
5. 保留上一版 lock、Catalog 和索引用于回滚。

active 指针指向包含 lock、Catalog、索引和报告的完整版本目录，该目录是运行时一致性的唯一事实来源；根部 `sources.lock.json` 和 `generated/` 是可由 active 版本重建的公开镜像。回滚恢复上一版完整集合，不允许只回滚索引或手工修改缓存。回滚后必须运行 active 版本一致性检查并重建公开镜像。

## 20. 构建报告

每次构建输出机器可读和人类可读报告，至少包含：

- 生成器与 Schema 版本；
- 各 Source 的请求 revision 和实际提交；
- 发现、启用、禁用、隔离和折叠数量；
- 新增、修改和删除 Skill；
- 重名、重复和 canonical 决策；
- 断裂、循环与越界引用；
- 索引模式和降级状态；
- 路由指标及相对基线变化；
- 是否通过发布门禁；
- active 版本与可回滚版本。

## 21. 当前仓库集成要求

实现时需要同步调整现有知识库入口：

1. `scripts/generate_llms_files.py` 不再依赖手工维护的完整 Skill 名单，而从有效 Catalog 动态生成 Skill 入口；
2. `llms.txt` 在靠前位置声明 Skill Routing Protocol，并链接紧凑目录；
3. `llms-full.txt` 包含路由协议与紧凑目录，但不盲目内联所有外部 Skill 正文；
4. 每个 Skill 正文和资源必须在知识摄取或 Resolver 中按稳定 ID 与锁定版本可寻址；
5. `skills/README.md` 说明本地 Skill 如何自动进入目录；
6. `AI_CONTINUATION_GUIDE.md` 加入强制维护协议；
7. `docs/knowledge-map.md` 和 MkDocs 导航加入路由系统及维护 Runbook；
8. 现有 `npu-arch-capability-check` 必须作为首个本地回归样例保留。

仅把 `catalog.jsonl` 作为普通文本喂给模型不足以实现按需加载。知识摄取系统必须同时保证 Catalog Entry 与对应 `SKILL.md` 可通过稳定标识解析；具体使用文件读取、RAG 文档检索或其他机制属于平台适配层，不改变本设计协议。

## 22. 验收场景

实现完成后必须通过以下端到端场景：

1. 新维护者只按 Runbook 新增一个本地 Skill，不修改 Router；目录自动发现，未知 Skill 名称的用户问题能够选中它。
2. 只在 `sources.yaml` 新增一个外部仓库；生成 lock 后，其有效 Skill 自动进入候选。
3. 同时启用 `agent-skills` 和 CANNBot；CANNBot 子树不产生重复候选。
4. 上游增加一个损坏 Skill；构建进入隔离并阻止未确认的新错误发布，active 目录保持可用。
5. 用户用自然语言描述 Ascend 环境检查但不提 Skill 名；系统选中正确 Skill，并在启用前告知 `ascendc-env-check` 名称和用途。
6. 一个 `confirm` Skill 被选中；未收到用户同意时 Handoff 被拒绝。
7. 用户拒绝已选 Skill；本次任务重新路由时不会再次选择它。
8. 没有合适 Skill；系统使用普通模型能力，不虚构 Skill。
9. 新 Skill 与旧 Skill 描述相近并错误抢占任务；路由回归评测阻止发布。
10. 语义索引不可用；系统报告降级并使用关键词检索完成既有回归样例。
11. 构建中途失败；active Catalog、lock 和索引完全不变。
12. AI 维护者只读维护文档即可完成一次来源更新、验证、发布或回滚，并能解释每个生成差异。

## 23. 实施边界

该设计适合拆成一个实现计划，但必须按依赖顺序交付：

1. 数据契约、Source Registry、Catalog Builder 与测试夹具；
2. 去重、Override、lock、隔离和事务式发布；
3. 关键词检索、可选语义检索与结构化 Selector；
4. Disclosure Gate、Lazy Loader、Handoff 和再路由状态；
5. 路由评测、质量门禁和构建报告；
6. 完整维护文档、AI 续作协议和现有 `llms` 索引集成。

任何阶段都不得用硬编码具体 Skill 名称替代通用数据契约。
