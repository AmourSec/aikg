# 自动发现、选择并使用 Skills：设计方案

**状态：** 设计完成，待评审后落地原型
**对应需求：** [AUTOMATIC_SKILL_ROUTING_TASK.md](AUTOMATIC_SKILL_ROUTING_TASK.md)

本文件回答“怎么做”，需求文档回答“要做什么”。两者保持对应，需求变更先改 TASK，设计变更先改本文件。

---

## 1. 设计目标复述

让用户用自然语言描述任务，系统自动完成：

1. 发现候选 Skills（不要求用户知道 Skill 名称）；
2. 语义判断适合的 Skills（可零个、一个、多个）；
3. 使用前向用户知会真实名称与用途，必要时等待确认；
4. 加载选中 Skills 的完整说明并继续执行。

核心约束（来自需求第 3、6 节）：

- 不绑定特定 Agent 平台、模型 API 或运行时；
- 路由逻辑不硬编码具体 Skill 名称；
- 新增 Skill 或新增来源不需要修改路由代码；
- Skills 数量增加后，不应每次把所有 Skill 全文塞进上下文。

---

## 2. 总体架构

分层解耦，每层职责单一，便于独立演进和替换。

```text
┌─────────────────────────────────────────────────────────────┐
│ Sources 层                                                   │
│   local 目录 / GitCode 仓库 / 其他可扫描目录                  │
│   每个 source 在 sources.yaml 里声明，无专用代码分支         │
└──────────────────────────┬──────────────────────────────────┘
                           │ build_catalog.py 扫描
                           │ **/SKILL.md 的 frontmatter
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Catalog 层（轻量目录 catalog.json）                          │
│   每条记录只含: name / description / source / path /         │
│   confirm / tags / triggers                                  │
│   不含 SKILL.md 正文，控制上下文体积                         │
└──────────────────────────┬──────────────────────────────────┘
                           │ 召回阶段: 关键词/语义粗筛
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Candidates 层（少量候选，3~10 条）                            │
│   仍是元数据，可附带 description 全文                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ 选择阶段: 大模型语义判断
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Selected 层（零个/一个/多个，可带顺序）                       │
└──────────────────────────┬──────────────────────────────────┘
                           │ 知会/确认
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Notify / Confirm 层                                          │
│   notify: 展示后可继续                                       │
│   confirm: 等待用户明确同意                                  │
│   任一选中 Skill 要求 confirm，整组等待                      │
└──────────────────────────┬──────────────────────────────────┘
                           │ 加载
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Execution 层                                                 │
│   按需读取选中 SKILL.md 全文 + references/scripts/assets     │
│   由大模型按 Skill 自身说明继续完成任务                      │
└─────────────────────────────────────────────────────────────┘
```

**关键不变量：**

- Catalog 层始终是轻量元数据，正文按需加载；
- 路由代码只认 Catalog schema，不认具体 Skill 名称；
- sources.yaml 是唯一来源真相，build_catalog.py 是通用扫描器。

---

## 3. Skill 元数据约定

为支持自动发现与策略判断，Skill 的 `SKILL.md` frontmatter 在现有约定（`name`、`description`，见 `docs/11-knowledge-index/skills-authoring-guide.md`）基础上，**可选**增加以下字段：

```yaml
---
name: npu-env-baseline
description: Use when asked to collect or verify Ascend NPU environment baseline, including CANN, driver, firmware, runtime, torch_npu, device model, SocVersion, NpuArch, container, and framework versions. This skill is for evidence collection before debugging or benchmarking, not for general NPU education.
tags: [npu, env, evidence]          # 可选，用于召回与策略分组
confirm: false                      # 可选，缺省由 sources.yaml defaults 决定
triggers:                           # 可选，显式触发线索，辅助召回
  - ascend
  - cann
  - npu-smi
---
```

**字段语义：**

| 字段 | 必填 | 作用 | 缺省值 |
| --- | --- | --- | --- |
| `name` | 是 | Skill 唯一标识，与目录名一致 | — |
| `description` | 是 | 召回与选择的主要依据，必须写清触发条件与边界 | — |
| `tags` | 否 | 召回辅助 + confirm 策略分组 | `[]` |
| `confirm` | 否 | 是否强制确认 | 由 `sources.yaml` 的 `defaults.confirm_tags` 推导 |
| `triggers` | 否 | 显式触发词，弥补 description 不够时召回 | 从 description 自动抽取 |

**向后兼容：** 现有 `skills/npu-arch-capability-check/SKILL.md` 只含 `name` 和 `description`，无需修改即可被发现。新增字段全部可选。

**元数据校验：** `build_catalog.py` 在扫描时校验：

- `name` 非空且与目录名一致；
- `description` 非空且长度 ≥ 20 字符（避免“NPU skill”这种无效描述）；
- `name` 在全 Catalog 内唯一（跨来源冲突时记录 conflict，不静默覆盖）；
- YAML 解析失败时记录 error，不中断整体扫描。

---

## 4. Sources 配置（sources.yaml）

唯一来源真相，维护者维护。新增来源只改此文件。

```yaml
# skills-router/config/sources.yaml
# 维护者手动维护。新增来源只需在 sources 下追加一条。

sources:
  - name: local
    type: local
    root: skills                       # 相对仓库根的目录
    description: 本仓库内置 skills

  - name: ascend-agent-skills
    type: git
    url: https://gitcode.com/Ascend/agent-skills
    branch: main
    root: ""                           # 仓库根，扫描 **/SKILL.md
    description: 昇腾 agent-skills 仓库
    sync_dir: .skills-cache/ascend-agent-skills   # 本地缓存路径

  - name: cannbot-skills
    type: git
    url: https://gitcode.com/cann/cannbot-skills
    branch: main
    root: ""
    description: CANN cannbot-skills 仓库
    sync_dir: .skills-cache/cannbot-skills

# 全局默认策略
defaults:
  confirm_policy: notify               # notify | confirm
  confirm_tags:                        # 命中任一 tag 的 skill 默认 confirm
    - destructive
    - write-fs
    - network
    - exec-shell
  description_min_chars: 20            # description 最小长度
```

**字段语义：**

- `type: local`：直接扫描本地目录，不做同步。
- `type: git`：先 `git clone`/`pull` 到 `sync_dir`，再扫描。两个 GitCode 仓库用同一种机制接入，无专用代码分支。
- `defaults.confirm_policy`：全局默认，单个 Skill 可在 frontmatter 覆盖。
- `defaults.confirm_tags`：tag 命中即升级为 confirm，便于按类别管控而无需逐个声明。

**为什么不用代码硬编码来源：** 需求第 6 节要求“新增一个 Skills 来源时，主要通过配置或目录接入，不增加仓库专用代码分支”。sources.yaml 满足。

---

## 5. Catalog 生成（build_catalog.py）

通用扫描器，读 sources.yaml，产 catalog.json。不依赖任何 Skill 名称。

**输入：** `skills-router/config/sources.yaml`
**输出：** `skills-router/catalog.json`

**catalog.json schema：**

```json
{
  "version": "1.0",
  "generated_at": "2026-08-03T10:00:00Z",
  "sources": [
    {"name": "local", "type": "local", "root": "skills"},
    {"name": "ascend-agent-skills", "type": "git", "url": "...", "synced_at": "..."}
  ],
  "skills": [
    {
      "name": "npu-arch-capability-check",
      "description": "Use when asked to judge Ascend NPU model, SocVersion, NpuArch, ...",
      "source": "local",
      "path": "skills/npu-arch-capability-check/SKILL.md",
      "tags": [],
      "triggers": ["ascend", "npu", "socversion", "npuarch"],
      "confirm": false
    }
  ],
  "conflicts": [],
  "errors": []
}
```

**扫描逻辑（伪代码）：**

```python
for source in load_sources_yaml():
    if source.type == "git":
        sync_git(source)                      # clone or pull to sync_dir
    for skill_md in glob(source.root, "**/SKILL.md"):
        meta = parse_frontmatter(skill_md)
        validate(meta, source)                # name/desc/唯一性
        triggers = meta.triggers or extract_keywords(meta.description)
        confirm = meta.confirm or any(t in defaults.confirm_tags for t in meta.tags)
        catalog.skills.append({...})
write_json(catalog, "skills-router/catalog.json")
```

**triggers 自动抽取：** 当 frontmatter 未声明 `triggers` 时，从 description 中抽取小写 token（去停用词），仅用于召回粗筛，不影响选择决策。大模型仍是最终判断者（满足需求第 5 节“不能只依赖关键词命中”）。

**conflicts / errors：**

- `conflicts`：同名 Skill 跨来源冲突。记录两条路径，路由时该名称不参与选择，并在 notify 时提示维护者。
- `errors`：YAML 解析失败、name 缺失、description 过短等。记录文件路径与原因，不中断扫描。

---

## 6. 路由协议（平台无关）

本层是“协议”而非“代码”，任何支持工具调用的 Agent 平台都可适配。完整文本见 `skills-router/ROUTING_PROTOCOL.md`，核心如下。

### 6.1 协议输入

```text
- catalog.json 内容（轻量，全量载入）
- 用户自然语言任务
```

### 6.2 协议步骤

```text
Step 1  RECALL（召回）
        输入: 用户任务 + catalog.skills[]
        规则: 关键词/triggers 命中 + description 语义相关
        输出: candidates[] (≤ 10 条，仍是元数据)
        约束: 宁可多召回，由 Step 2 精筛

Step 2  SELECT（选择）
        输入: candidates[] + 用户任务
        规则: 大模型语义判断，可返回 0/1/N 个
        输出: selected[]，带 order 字段表示使用顺序
        约束: 不得选择 candidates 之外的 skill；无匹配时返回空数组

Step 3  NOTIFY / CONFIRM
        输出固定格式消息:
          准备使用以下 Skills：
          - <name>：<一句话用途>（confirm）
          - <name>：<一句话用途>
        confirm 判定: 任一 selected.confirm == true → 整组等待
        notify: 展示后可继续
        confirm: 必须等待用户明确同意（"继续"/"yes"）后才能进入 Step 4

Step 4  LOAD（加载）
        对每个 selected，按 path 读取 SKILL.md 全文
        若 SKILL.md 引用 references/scripts/assets，按需读取
        约束: 只加载 selected，不加载全 Catalog 正文

Step 5  EXECUTE
        大模型按 Skill 自身说明继续完成任务
        若 Skill 内部要求子步骤确认，由 Skill 自身约定
```

### 6.3 选择输出格式（JSON）

```json
{
  "selected": [
    {"name": "npu-env-baseline", "order": 1, "reason": "任务涉及环境基线收集"},
    {"name": "npu-arch-capability-check", "order": 2, "reason": "需要判断 NpuArch 是否支持目标算子"}
  ],
  "rejected": [
    {"name": "inference-benchmark-pack", "reason": "任务不涉及 benchmark"}
  ],
  "confirm_required": true,
  "confirm_reason": "npu-env-baseline 标 confirm（exec-shell）"
}
```

### 6.4 无匹配时的输出

```json
{"selected": [], "rejected": [], "confirm_required": false}
```

并附自然语言说明：未找到适合的 Skill，将直接基于知识库回答。

### 6.5 平台适配参考

协议本身平台无关。具体平台只需实现一个“工具”把上述步骤暴露给大模型：

- **opencode**：可用 `skill` 工具加载，用 `question` 工具实现 confirm；catalog 作为系统提示注入。参考实现见 `skills-router/README.md` 的“opencode 适配”一节。
- **OpenAI Function Calling**：定义 `recall_skills`、`select_skills`、`load_skill` 三个函数。
- **纯 Prompt Agent**：把 catalog 和协议步骤写进 system prompt，让模型直接输出 JSON。

---

## 7. Notify / Confirm 机制

### 7.1 两种策略

| 策略 | 行为 | 适用 |
| --- | --- | --- |
| `notify` | 展示 Skill 名称与用途后可继续 | 只读、低风险 skill |
| `confirm` | 必须等待用户明确同意 | 写文件、执行命令、网络请求等 |

### 7.2 confirm 触发条件（任一满足）

1. Skill frontmatter 显式声明 `confirm: true`；
2. Skill `tags` 命中 `sources.yaml` 的 `defaults.confirm_tags`；
3. 来源仓库策略要求（可在 sources.yaml 单个 source 上加 `default_confirm: true`）。

### 7.3 整组等待原则

多个 Skills 同时使用时，任一要求 confirm，整组都等待（满足需求第 4.2 节）。用户同意后，后续同组 Skills 不再重复 confirm。

### 7.4 用户拒绝时的行为

- 用户拒绝某个 confirm Skill：该 Skill 不加载，路由协议重新评估剩余 selected 是否仍可完成任务；
- 若剩余 Skill 无法独立完成任务：向用户说明缺哪个 Skill 会阻塞哪一步，并询问是否调整任务范围。

### 7.5 confirm 不依赖平台原生机制

平台无关实现：协议规定“模型输出 confirm 消息后必须停止生成，等待下一轮用户输入”。任何 Agent 平台只要支持“模型停止 → 用户输入 → 模型继续”的回合，就能实现 confirm。

---

## 8. 文件布局

```text
automatic-skill-routing/
  README.md                          # 总说明
  OVERVIEW.md                        # 快速理解
  AUTOMATIC_SKILL_ROUTING_DESIGN.md  # 本文件
  skills-router/
    README.md                        # 维护文档（新人/AI 可据此操作）
    ROUTING_PROTOCOL.md              # 平台无关路由协议
    opencode-adapter.md              # opencode 平台适配参考
    config/
      sources.yaml                   # 维护者维护，来源真相
    scripts/
      build_catalog.py               # 通用扫描器，生成 catalog.json
      validate_catalog.py            # 校验 catalog 完整性与一致性
      generate_router_context.py     # 生成可注入的 router-context.md
      test_routing.py                # 回归测试
    catalog.json                     # 自动生成，git 可跟踪以便审计
    router-context.md                # 自动生成，注入模型上下文用
```

**人维护 vs 自动生成：**

| 文件 | 维护者 | 自动生成 |
| --- | --- | --- |
| `config/sources.yaml` | 是 | — |
| `catalog.json` / `router-context.md` | — | 是（build_catalog.py / generate_router_context.py） |
| `README.md` / `ROUTING_PROTOCOL.md` | 是 | — |
| 各 source 的 `SKILL.md` | 是（Skill 作者） | — |
| `scripts/*.py` | 是（本系统维护者） | — |

---

## 9. 扩展性验证（对应需求第 6 节）

| 需求 | 本设计如何满足 |
| --- | --- |
| 路由逻辑不硬编码业务 Skill 名称 | build_catalog.py 只扫 `**/SKILL.md`，路由协议只认 catalog schema |
| 新增合法 Skill 即可被发现 | 放到任一 source 目录，重跑 build_catalog 即进 catalog |
| 新增来源不增加专用代码分支 | sources.yaml 加一条，type=git 自动 clone 扫描 |
| 元数据来自 Skill 自身 | frontmatter 字段，无外部登记表 |
| 不把所有 Skill 全文塞进上下文 | catalog.json 只含元数据，正文 Step 4 按需加载 |
| 先轻量目录再按需读取 | Catalog → Candidates → Selected → Load 四阶段 |

---

## 10. 验收标准对应（对应需求第 9 节）

| # | 验收项 | 设计落点 |
| --- | --- | --- |
| 1 | 用户不说 Skill 名称也能找到 | Step 1 召回基于任务语义，不要求用户指定 |
| 2 | 无关任务不强行分配 | Step 2 允许返回空 selected[] |
| 3 | 多步骤任务选有顺序的多个 | selected[] 带 order 字段 |
| 4 | 使用前展示真实名称与用途 | Step 3 NOTIFY 固定格式 |
| 5 | confirm 的 Skill 同意前不使用 | Step 3 整组等待 + Step 4 才加载 |
| 6 | 新增测试 Skill 无需改路由代码 | build_catalog 通用扫描，sources.yaml 不改即可发现 local 目录新 Skill |
| 7 | 新增来源不增加专用分支 | sources.yaml type=git 通用机制 |
| 8 | 两个 GitCode 仓库同种机制接入 | sources.yaml 中 ascend-agent-skills 与 cannbot-skills 配置结构相同 |
| 9 | 核心流程不依赖特定平台 | ROUTING_PROTOCOL.md 平台无关，opencode 仅作为适配参考 |
| 10 | 新人或 AI 能据文档完成操作 | README.md 覆盖新增/更新/验证/故障恢复 |

---

## 11. 落地状态

本轮已**端到端落地**，包括：

- 实际同步两个 GitCode 仓库（`Ascend/agent-skills` 196 skills、`cann/cannbot-skills` 189 skills）；
- 生成完整 catalog（386 skills，284 KB）；
- 生成可注入路由上下文 `router-context.md`（165 KB）；
- 回归测试 6 项全部通过；
- 平台无关，大模型自适应执行协议。

5 个同名冲突和 9 个格式错误来自远程仓库内部，系统已正确识别并排除，不影响可用 Skills。

不包含（边界外）：

- 不引入向量数据库（召回阶段用关键词 + 大模型语义，386 skills 规模下有效）；
- 不建设通用工作流编排平台（边界见需求第 3 节）；
- 不绑定特定 Agent 平台（协议平台无关，大模型自适应）。

---

## 12. 决策点确认结果

以下决策点已由用户确认并落地：

| 决策点 | 确认结果 | 落地状态 |
| --- | --- | --- |
| 交付范围 | 端到端落地 | ✅ 完成 |
| 知识库提供形式 | 本仓库 docs/ + skills/ | ✅ 已有 |
| 是否同步远程仓库 | 本轮启用 | ✅ 两个 GitCode 仓库已同步 |
| 默认 confirm 类别 | destructive / write-fs / network / exec-shell | ✅ 在 sources.yaml |
| 是否提供 CLI | 是，4 个脚本 | ✅ build/validate/generate/test |
| 交付目录 | `automatic-skill-routing/` | ✅ 自包含 |
| 平台绑定 | 平台无关，大模型自适应 | ✅ 协议 + router-context.md |
| 是否创建分支/提交 | 新建分支并提交 | ✅ feat/automatic-skill-routing |
