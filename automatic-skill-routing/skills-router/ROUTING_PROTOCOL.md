# Skills Routing Protocol

**平台无关的路由协议。** 任何支持“工具调用 + 多轮对话”的 Agent 平台都可适配。本文件不绑定 opencode、OpenAI 或任何特定框架。

协议对应 `AUTOMATIC_SKILL_ROUTING_TASK.md` 第 4、5 节的交互与选择要求，以及 `AUTOMATIC_SKILL_ROUTING_DESIGN.md` 第 6 节的架构。

---

## 1. 协议输入

每次会话开始时，Agent 应获得：

1. 本协议全文（作为系统提示的一部分）；
2. `skills-router/catalog.json` 的 `skills` 数组（轻量元数据，全量载入）；
3. `sources.yaml` 的 `defaults` 段（用于 confirm 策略说明）。

**不载入**任何 SKILL.md 正文。正文在 Step 4 按需加载。

---

## 2. 协议步骤

### Step 1 — RECALL（召回）

**输入：** 用户自然语言任务 + catalog.skills[]
**规则：**
- 关键词匹配：用户任务文本与 skill 的 `triggers` / `tags` / `name` / `description` 做大小写不敏感匹配；
- 语义相关：description 与任务语义相关（由大模型判断）；
- 宁可多召回，由 Step 2 精筛。
**输出：** candidates[]，≤ 10 条，仍是元数据。
**约束：** 不输出 catalog 之外的 skill。

### Step 2 — SELECT（选择）

**输入：** candidates[] + 用户任务
**规则：**
- 大模型语义判断每个 candidate 是否真正适合任务；
- 可返回 0、1 或 N 个 selected；
- 多个 skill 时按 `order` 字段标注使用顺序；
- 必须为每个 selected 给出 `reason`，为每个被否决的 candidate 给出 `reason`。
**输出格式（JSON）：**

```json
{
  "selected": [
    {"name": "<skill-name>", "order": 1, "reason": "<为何适合>"},
    {"name": "<skill-name>", "order": 2, "reason": "<为何适合>"}
  ],
  "rejected": [
    {"name": "<skill-name>", "reason": "<为何不适合>"}
  ],
  "confirm_required": false,
  "confirm_reason": ""
}
```

**约束：**
- `selected` 中的 `name` 必须在 candidates 之内；
- 无匹配时 `selected` 为空数组，并附自然语言说明“未找到适合的 Skill，将直接基于知识库回答”；
- `confirm_required` = `selected` 中任一 skill 的 `confirm == true`；
- 若 `confirm_required` 为 true，`confirm_reason` 必须说明哪个 skill 触发了 confirm 及原因。

### Step 3 — NOTIFY / CONFIRM

无论 selected 为空、一个或多个，都先输出固定格式的知会消息。

**notify（任一 skill 都不要求 confirm）：**

```text
准备使用以下 Skills：
- <name>：<一句话用途>
- <name>：<一句话用途>

如需调整，请说明；否则我将继续。
```

展示后可继续进入 Step 4。

**confirm（任一 skill 要求 confirm）：**

```text
准备使用以下 Skills：
- <name>：<一句话用途>（需确认：exec-shell）
- <name>：<一句话用途>

其中 <name> 需要您明确同意后才能使用。请回复“继续”或“yes”以确认，或说明如何调整。
```

输出此消息后，**模型必须停止生成，等待用户下一轮输入**。在用户明确同意前，不得进入 Step 4。

**整组等待原则：** 多个 skill 同时使用时，任一要求 confirm，整组等待。用户同意后，同组后续 skill 不再重复 confirm。

**用户拒绝时：**
- 拒绝某个 confirm skill：该 skill 从 selected 移除，重新评估剩余 selected 是否仍可完成任务；
- 若剩余 skill 无法独立完成：说明缺哪个 skill 会阻塞哪一步，并询问是否调整任务范围。

### Step 4 — LOAD（加载）

对每个 selected skill，按 `path` 读取 SKILL.md 全文。

- 若 SKILL.md 引用 `references/`、`scripts/`、`assets/`，按需读取引用的文件；
- 只加载 selected，不加载全 catalog 正文；
- 加载失败（文件缺失、权限问题）时记录 error 并告知用户，不静默跳过。

### Step 5 — EXECUTE

大模型按 Skill 自身说明继续完成任务。

- 若 Skill 内部要求子步骤确认（例如执行命令前确认），由 Skill 自身约定，不由本协议强制；
- 任务完成后，可附简要说明使用了哪些 skill。

---

## 3. 召回规则细节

召回阶段使用以下信号，按优先级递减：

1. `triggers` 显式命中用户任务文本（最强信号）；
2. `tags` 命中用户任务关键词；
3. `name` 子串命中；
4. `description` 语义相关（大模型判断）。

**关键词命中不等于选择。** 召回只负责把候选缩小到 ≤ 10 条，最终是否使用由 Step 2 大模型语义判断（满足需求第 5 节“不能只依赖关键词命中”）。

---

## 4. confirm 策略细节

### 触发条件（任一满足即 confirm）

1. Skill frontmatter 显式 `confirm: true`；
2. Skill `tags` 命中 `sources.yaml` 的 `defaults.confirm_tags`（默认：`destructive` / `write-fs` / `network` / `exec-shell`）；
3. 来源仓库在 sources.yaml 单个 source 上声明 `default_confirm: true`。

### 默认 confirm_tags 语义

| tag | 含义 |
| --- | --- |
| `destructive` | 可能删除或覆盖数据 |
| `write-fs` | 写文件系统 |
| `network` | 发起网络请求 |
| `exec-shell` | 执行 shell 命令 |

新增 tag 只需在 sources.yaml 追加，无需改协议或代码。

### confirm 不依赖平台原生机制

平台无关实现：协议规定“模型输出 confirm 消息后必须停止生成，等待下一轮用户输入”。任何 Agent 平台只要支持“模型停止 → 用户输入 → 模型继续”的回合，就能实现 confirm。

---

## 5. 无匹配行为

当 Step 2 返回空 selected 时：

```text
未找到适合当前任务的 Skill。我将直接基于知识库内容回答。

如果您认为应该有合适的 Skill，可以：
- 描述更具体的任务场景；
- 检查 skills-router/catalog.json 是否需要重新生成。
```

不得勉强匹配（满足需求第 5 节“没有适用 Skill 时允许返回‘不使用 Skill’”）。

---

## 6. 平台适配参考

协议本身不依赖任何平台。具体平台只需把 Step 1–5 映射到平台能力：

### 6.1 纯 Prompt Agent（最简）

把本协议 + catalog.json 写进 system prompt，让模型直接输出 Step 2 的 JSON，再按 Step 3 输出知会消息。

### 6.2 OpenAI Function Calling

定义三个函数：

- `recall_skills(task: str) -> list[dict]`：返回 candidates；
- `select_skills(candidates: list[dict], task: str) -> dict`：返回 Step 2 的 JSON；
- `load_skill(name: str) -> str`：读取 SKILL.md 全文。

confirm 通过“模型输出消息后停止生成”实现，无需专用函数。

### 6.3 opencode

opencode 已内置 `skill` 工具用于加载 skill，`question` 工具可用于 confirm 交互。catalog 作为系统提示注入。具体适配见 `opencode-adapter.md`。

---

## 7. 协议不变量

以下不变量在任何平台适配中都必须保持：

1. **不硬编码 skill 名称**：路由逻辑只认 catalog schema，不认具体 skill；
2. **catalog 之外不选**：selected 必须在 candidates 之内，candidates 必须在 catalog 之内；
3. **正文按需加载**：只加载 selected 的 SKILL.md，不加载全 catalog 正文；
4. **confirm 整组等待**：任一 selected.confirm == true，整组等待用户同意；
5. **无匹配允许**：selected 可为空，不勉强匹配；
6. **知会前置**：Step 4 加载前必须完成 Step 3 知会。

违反任一不变量的适配视为不合规。
