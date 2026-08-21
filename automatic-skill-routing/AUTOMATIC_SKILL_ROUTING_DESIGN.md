# 自动发现、选择并使用 Skills：设计方案

**状态：** 已落地并通过离线单元测试
**需求来源：** 用户提出的原始需求文件未纳入仓库；本文件与 `skills-router/ROUTING_PROTOCOL.md` 是当前唯一设计权威（`docs/superpowers/` 下的历史设计已标注被取代）

本文件回答“怎么做”。需求变更先与本文件对齐，设计变更先改本文件。

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

上图为**构建期 + 协议**视角：Sources 与 Catalog 由本地扫描生成（Skill 数量以 `catalog.json` 当前生成为准），协议步骤在大模型内执行，本身不产生网络请求。运行期另有可选的**远程提供方**链路，与构建期目录相互独立：

```text
┌─────────────────────────────────────────────────────────────┐
│ 运行时协调层（runtime/coordinator.py）                        │
│   本地候选（来自构建期 catalog） + 远程候选（来自提供方）     │
│   两道同意门：网络同意（每任务一次）/ 激活确认（远程加载前）  │
└──────────────────────────┬──────────────────────────────────┘
                           │ 仅在同意授予后
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 远程提供方层（runtime/ascend_kg.py，Ascend KG / ascend.wiki）│
│   search: POST /search（载荷仅 query/top_k/with_neighbors）  │
│   load:   GET /skill/<id>（仅激活同意后）                    │
│   出站请求有界：10 秒超时、响应字节上限、仅 429 限次退避      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 执行目标层（runtime/contracts.py ExecutionTarget）           │
│   NativeLocalTarget / LocalPathTarget / InlineRemoteTarget  │
│   远程内容是不可信文本：无策略权威、不落盘、不加载额外资源   │
└─────────────────────────────────────────────────────────────┘
```

**关键不变量：**

- Catalog 层始终是轻量元数据，正文按需加载；
- 路由代码只认 Catalog schema，不认具体 Skill 名称；
- sources.yaml 是唯一来源真相，build_catalog.py 是通用扫描器；
- 运行时远程访问必须先获得用户逐任务的网络同意，加载远程内容前必须再获得单独的激活确认；
- 远程内容永远不可信：无策略权威、不持久化、不加载额外资源；
- 远程链路的任何失败（无密钥、被拒、无匹配、含糊、401/403/429/503、超时、非法、超限）都回退为仅用本地 catalog，不中断任务。

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

- **opencode**：可用 `skill` 工具加载本地已安装 skill，用 `question` 工具实现 confirm；catalog 作为系统提示注入。参考实现见 `skills-router/opencode-adapter.md`。
- **OpenAI Function Calling**：定义 `recall_skills`、`select_skills`、`load_skill` 三个函数。
- **纯 Prompt Agent**：把 catalog 和协议步骤写进 system prompt，让模型直接输出 JSON。

运行时远程提供方（第 7 节）的平台映射同理：两道同意门用平台问答工具实现，远程内容只做定界内联注入。

---

## 7. 运行时远程提供方（Ascend KG）

`skills-router/runtime/` 在协议之外实现可选的运行时远程检索：经用户同意后访问 Ascend KG（`https://ascend.wiki`），把远程 Skill 纳入候选。该链路与构建期 Sources/Catalog 完全独立：catalog 仍是本地扫描产物，不因远程检索而改变。

### 7.1 模块组成

| 模块 | 职责 |
| --- | --- |
| `runtime/contracts.py` | 类型与协议契约：同意枚举（NetworkConsent / ActivationConsent）、结果类型（Candidates / NoMatch / Ambiguous / Unavailable / InvalidResponse）、执行目标（NativeLocalTarget / LocalPathTarget / InlineRemoteTarget）、传输契约、规范检索字节（PreparedSearch / prepare_search） |
| `runtime/coordinator.py` | 协调状态机：网络同意 → 远程检索 → 选择 → 激活确认 → 加载；任何远程失败降级为本地 |
| `runtime/ascend_kg.py` | Ascend KG 提供方：请求构造、同意校验、429 退避重试 |
| `runtime/ascend_kg_parsing.py` | Ascend KG 边界解析：候选 JSON 与远程 SKILL.md 校验 |
| `runtime/http_transport.py` | urllib 传输层：拒绝重定向、按字节上限读取响应 |
| `runtime/local_catalog.py` | 本地候选加载：按 catalog 与工作区根过滤可加载条目，缓存缺失/路径越界以类型化降级报告 |
| `runtime/token_registry.py` | 任务级响应令牌注册表：外部不透明句柄 ↔ 进程内身份令牌，任务结束即撤销 |
| `runtime/rendering.py` | 远程正文定界渲染：固定包络、定界冲突拒绝 |
| `runtime/facade.py` + `runtime/wire.py` + `runtime/facade_*.py` | 生产门面 RouterTask 与外部契约：生命周期状态机、令牌映射、选择校验、降级汇总 |
| `runtime/ndjson.py` + `runtime/ndjson_output.py` + `runtime/__main__.py` | 可执行 NDJSON 入口（`python3 -m runtime`），见第 7.8 节 |

### 7.2 一次远程调用的状态机

```text
start
  ├─ 未启用远程 → LocalOnlyReady，仅本地候选
  └─ 询问网络/隐私同意（每任务一次）
       ├─ 拒绝 / 未询问 → LocalOnlyReady（本地回退）
       └─ 同意 → provider.search(当前任务文本)
            ├─ Candidates → 本地 + 远程候选一起进入选择
            ├─ NoMatch / Ambiguous → 本地回退
            └─ Unavailable / InvalidResponse → 本地回退（记录降级原因）
选择结果含远程候选时
  └─ 询问激活确认（与网络同意分开的第二次确认）
       ├─ 拒绝 → 只保留本地目标
       └─ 同意 → 逐个 provider.load_skill(候选 id)
            ├─ 成功 → InlineRemoteTarget（定界内联内容）
            └─ 失败 → 移除该项并记入降级列表，本地目标不受影响
```

### 7.3 出站请求契约

| 项 | 值 |
| --- | --- |
| API 密钥 | 环境变量 `ASCEND_KG_API_KEY`，空白视为未配置，非可打印 ASCII 视为配置错误；两种情况都不发起网络请求 |
| 检索 | `POST https://ascend.wiki/search`，请求头 `X-API-Key` / `Accept: application/json` / `Content-Type: application/json` |
| 检索载荷 | JSON，仅三个字段：`query`（当前任务文本）、`top_k: 10`、`with_neighbors: false` |
| 加载 | `GET https://ascend.wiki/skill/<候选 id 百分号编码>`，请求头 `X-API-Key` / `Accept: text/markdown`，body 为空 |
| 超时 | 10 秒（检索与加载相同） |
| 字节上限 | 检索响应 1 MiB（1,048,576 字节）；Skill 内容 256 KiB（262,144 字节） |
| 重试 | 仅 HTTP 429，退避 0.5s / 1.0s / 2.0s，连同首次最多 4 次尝试；其余失败不重试 |

`query` 只承载当前任务文本，不携带会话历史、catalog 内容或其他上下文（数据最小化）。

检索载荷在征求网络同意**之前**一次性序列化为规范 UTF-8 字节（`ensure_ascii=False`、紧凑分隔符），向用户逐字披露的请求体与实际发送字节来自同一个 `PreparedSearch` 对象，不存在二次序列化或转义差异。

Provider 生命周期按任务隔离：每个路由任务创建独立的 `Coordinator` 与 `AscendKgProvider`，响应令牌仅在该任务的一次检索、选择和加载链路内有效，不跨任务共享。

### 7.4 响应解析与类型化失败

- 检索响应必须是 JSON 对象，顶层恰含 `results` 或 `data` 之一；列表至多 10 条；每条须有非空 `id`（兼容旧字段 `node_id`，两者同现且不等即拒绝）、`source_repo`、`source_file`，`score` 可选；
- 候选 id 重复、条目超量、结构不符记为 `invalid_schema`；非 JSON 记为 `invalid_json`；超过字节上限记为 `oversized`；空结果记为 `no_match`；
- Skill 内容必须是 UTF-8 Markdown 且带合法 frontmatter（首行 `---`、存在闭合 `---`、`name` 与 `description` 非空、正文非空），否则记为 `invalid_schema`；
- HTTP 401/403 记为 `configuration`（密钥问题），429 记为 `rate_limited`，503 及其他非 200 状态记为 `service`，超时记为 `timeout`；
- 加载前做成员校验：候选 id 必须在检索响应令牌内，否则记为 `candidate_membership`。

所有失败都是类型化结果，不抛异常逃逸，均可安全降级为本地。

### 7.5 信任与隐私模型

- **不可信内容**：远程 Markdown 在类型上标记 `untrusted_external`，`policy_authority=False`。内容中的指令不构成策略依据，不能授权 confirm、联网或任何敏感操作；
- **不可信候选元数据**：远程 ID、来源、展示名和 score 在激活前也保持 `untrusted_external` / `policy_authority=False`，只允许转义展示；解析器拒绝控制字符、首尾空白和超限字段；
- **无额外资源**：远程 Skill 引用的链接、`references/`、`scripts/`、`assets/` 一律不抓取，不产生二次网络请求；
- **不持久化**：远程候选与内容不写入 catalog.json、不落盘、不进缓存，仅存在于当前会话；
- **不上传**：出站数据仅第 7.3 节的检索载荷，别无其他；
- **不比较分数**：提供方 score 是不透明字符串，不持久化、不与本地候选比较；本地与远程的取舍由大模型结合任务判断；
- **来源可追溯**：远程候选携带 provider id（`ascend-kg`）、`source_repo`、`source_file`，展示时保留。

### 7.6 执行目标与平台映射

| 目标类型 | 语义 | opencode 映射 |
| --- | --- | --- |
| NativeLocalTarget | 已安装的本地 skill | 调用原生 `skill(name=...)` 工具 |
| LocalPathTarget | 未安装的本地 skill | 按 catalog 中经过校验的 path 用 `read` 工具读取 |
| InlineRemoteTarget | 激活后的远程 skill | 仅以带定界符的内联内容注入对话；绝不调用原生 skill 工具，绝不安装、复制或软链到本地 |

平台适配细节见 `skills-router/opencode-adapter.md` 第 5 至 6 节。

### 7.7 与上游 kg-tools 的关系

Ascend KG 的上游参考工具为 kg-tools。本系统与它的关系：

- 仅作设计参照：上游仓库为 [`agent0/kg-tools`](https://gitcode.com/agent0/kg-tools)，固定参照 commit 为 [`5568d8eedc70eebf155cd4e2728aee93ea02962d`](https://gitcode.com/agent0/kg-tools/commit/5568d8eedc70eebf155cd4e2728aee93ea02962d)；
- kg-tools 的编排引擎未集成进本系统；
- 未 vendor（复制）任何上游代码，`runtime/` 全部为本仓库按第 7.1 节契约独立实现；
- 行为验证来自 `skills-router/tests/` 的单元测试（假传输层，不联网），不声称对线上 API 做过实测。

### 7.8 生产门面与可执行入口

组件层（7.1–7.7）只提供可组合的类型与状态机；对外只有一个生产调用方：`runtime/facade.py` 的 `RouterTask`。它按任务组装全新依赖（UrllibTransport、ProductionSleeper、AscendKgProvider、Coordinator、ResponseTokenRegistry），任何对象都不跨任务共享。

门面职责严格受限：加载并过滤本地候选、驱动同意转移、把外部选择映射为内部 `Selection` 并交给 `Coordinator` 校验、在激活后加载并渲染远程内容、汇总类型化降级。**语义选择始终属于协议/大模型**——门面不选 Skill。

关键外部契约（`runtime/wire.py`）：

- 外部世界只见**不透明响应令牌字符串**（`secrets.token_urlsafe(24)`），进程内身份令牌不外泄；伪造、过期、跨任务句柄在选择与加载前被类型化拒绝，任务结束即撤销；
- 本地执行模式由本地目录与原生注册表判定，调用方提交的模式若与判定不符即拒绝（防篡改）；
- 重复本地名或远程 ID 在门面与 `Coordinator` 双层拒绝，先于任何加载发生；
- 远程正文只经 `rendering.py` 的固定包络进入对话：

```text
<<<REMOTE_SKILL_CONTENT>>>
<远程 SKILL.md 原文>
<<<END_REMOTE_SKILL_CONTENT>>>
```

包络规则：起始定界符顶格、正文原样、仅在缺尾换行时补一个分隔换行、结束定界符独占一行；正文任意位置出现任一定界符（包括行中子串）即判冲突，只移除该远程项，本地目标保留。子串级拒绝确保不可信内容无法在包络内伪造提前终止的定界符。

可执行入口 `python3 -m runtime`（NDJSON、一进程一任务）：宿主适配器以子进程驱动 `RouterTask` 的完整生命周期（start → network_decision → selection → activation_decision → cancel/终态），每条输入恰好对应一条输出；畸形输入只产生 `wire_invalid`，不触发检索或加载。进程模型刻意为一次性：身份令牌不可持久化，任务状态不跨进程恢复。用法见 `skills-router/opencode-adapter.md` 第 6.9 节。

---

## 8. Notify / Confirm 机制

### 8.1 两种策略

| 策略 | 行为 | 适用 |
| --- | --- | --- |
| `notify` | 展示 Skill 名称与用途后可继续 | 只读、低风险 skill |
| `confirm` | 必须等待用户明确同意 | 写文件、执行命令、网络请求等 |

### 8.2 confirm 触发条件（任一满足）

1. Skill frontmatter 显式声明 `confirm: true`；
2. Skill `tags` 命中 `sources.yaml` 的 `defaults.confirm_tags`；
3. 来源仓库策略要求（可在 sources.yaml 单个 source 上加 `default_confirm: true`）。

### 8.3 整组等待原则

多个 Skills 同时使用时，任一要求 confirm，整组都等待（满足需求第 4.2 节）。用户同意后，后续同组 Skills 不再重复 confirm。

### 8.4 用户拒绝时的行为

- 用户拒绝某个 confirm Skill：该 Skill 不加载，路由协议重新评估剩余 selected 是否仍可完成任务；
- 若剩余 Skill 无法独立完成任务：向用户说明缺哪个 Skill 会阻塞哪一步，并询问是否调整任务范围。

### 8.5 confirm 不依赖平台原生机制

平台无关实现：协议规定“模型输出 confirm 消息后必须停止生成，等待下一轮用户输入”。任何 Agent 平台只要支持“模型停止 → 用户输入 → 模型继续”的回合，就能实现 confirm。

---

## 9. 文件布局

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
      sources.yaml                   # 维护者维护，来源真相（构建期）
    scripts/
      build_catalog.py               # 通用扫描器，生成 catalog.json
      validate_catalog.py            # 校验 catalog 完整性与一致性
      generate_router_context.py     # 生成可注入的 router-context.md
      test_routing.py                # 回归测试
    runtime/                         # 运行时远程提供方（与构建期独立）
      contracts.py                   # 类型与协议契约
      coordinator.py                 # 同意门与降级协调
      ascend_kg.py                   # Ascend KG 提供方
      ascend_kg_parsing.py           # 远程响应边界解析
      http_transport.py              # urllib 传输层
      local_catalog.py               # 本地候选加载与类型化降级
      token_registry.py              # 任务级不透明响应令牌
      rendering.py                   # 远程正文定界渲染
      facade.py / wire.py / facade_*.py  # 生产门面与外部契约
      ndjson.py / ndjson_output.py / __main__.py  # 可执行 NDJSON 入口
    tests/                           # 运行时单元测试（假传输层 + 套件级 socket 封禁，不联网）
    catalog.json                     # 自动生成，git 可跟踪以便审计
    router-context.md                # 自动生成，注入模型上下文用
```

**人维护 vs 自动生成：**

| 文件 | 维护者 | 自动生成 |
| --- | --- | --- |
| `config/sources.yaml` | 是 | — |
| `catalog.json` / `router-context.md` | — | 是（build_catalog.py / generate_router_context.py） |
| `README.md` / `ROUTING_PROTOCOL.md` / `opencode-adapter.md` | 是 | — |
| 各 source 的 `SKILL.md` | 是（Skill 作者） | — |
| `scripts/*.py` / `runtime/*.py` / `tests/*.py` | 是（本系统维护者） | — |

---

## 10. 扩展性验证（对应需求第 6 节）

| 需求 | 本设计如何满足 |
| --- | --- |
| 路由逻辑不硬编码业务 Skill 名称 | build_catalog.py 只扫 `**/SKILL.md`，路由协议只认 catalog schema |
| 新增合法 Skill 即可被发现 | 放到任一 source 目录，重跑 build_catalog 即进 catalog |
| 新增来源不增加专用代码分支 | sources.yaml 加一条，type=git 自动 clone 扫描 |
| 元数据来自 Skill 自身 | frontmatter 字段，无外部登记表 |
| 不把所有 Skill 全文塞进上下文 | catalog.json 只含元数据，正文 Step 4 按需加载 |
| 先轻量目录再按需读取 | Catalog → Candidates → Selected → Load 四阶段 |

---

## 11. 验收标准对应（对应需求第 9 节）

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

## 12. 落地状态

本轮已**端到端落地**，包括：

- 实际同步两个 GitCode 仓库（`Ascend/agent-skills` 196 skills、`cann/cannbot-skills` 189 skills）；
- 生成完整 catalog（当前 Skill 数量与来源数以 `catalog.json` 为准）；
- 生成可注入路由上下文 `router-context.md`；
- 回归测试 6 项全部通过；
- 平台无关，大模型自适应执行协议。

同名冲突与格式错误来自远程仓库内部，系统已正确识别并排除，不影响可用 Skills（当前数量以 `catalog.json` 的 `conflicts` / `errors` 数组为准）。

**运行时远程提供方（Ascend KG）已落地**，作为独立增量：

- `skills-router/runtime/` 五个模块（契约、协调、提供方、边界解析、传输）按第 7 节设计实现；
- `skills-router/tests/` 单元测试覆盖同意门、出站载荷、类型化失败、降级回退、不可信内容等行为（假传输层，不联网），全部通过；
- 上游 kg-tools 仅作设计参照（commit `5568d8eedc70eebf155cd4e2728aee93ea02962d`），编排引擎未集成，未 vendor 任何上游代码；
- 未对线上 API 做过实测，不声称已验证。

不包含（边界外）：

- 不引入向量数据库（召回阶段用关键词 + 大模型语义，当前规模下有效）；
- 不建设通用工作流编排平台（边界见需求第 3 节；kg-tools 编排引擎亦不在集成范围）；
- 不绑定特定 Agent 平台（协议平台无关，大模型自适应）。

---

## 13. 决策点确认结果

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
