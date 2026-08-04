# opencode 平台适配参考

**本文件是参考实现，不是协议必需。** 协议本身见 `ROUTING_PROTOCOL.md`，平台无关。本文件说明如何在 opencode 中落地协议，便于直接使用或作为其他平台适配的对照。

---

## 1. opencode 已有能力映射

| 协议步骤 | opencode 能力 | 说明 |
| --- | --- | --- |
| 协议输入 | 系统提示 | 把 `ROUTING_PROTOCOL.md` + `catalog.json` 的 skills 数组写进系统提示 |
| Step 1 RECALL | 模型推理 | 模型基于 catalog 元数据自行召回 |
| Step 2 SELECT | 模型输出 JSON | 模型按协议输出 selected/rejected JSON |
| Step 3 NOTIFY | 文本输出 | 模型直接输出固定格式消息 |
| Step 3 CONFIRM | `question` 工具 | 用 `question` 工具询问用户是否继续 |
| Step 4 LOAD | `skill` 工具 | 用 `skill` 工具加载选中 skill |
| Step 5 EXECUTE | 模型按 skill 说明继续 | skill 加载后，模型按其内容执行 |

---

## 2. 最小适配：纯 Prompt 模式

无需任何代码。在 opencode 的系统提示或 `AGENTS.md` 中加入：

```markdown
## Skills Routing

当用户提出任务时，按以下协议选择并使用 Skills：

1. 读取 skills-router/catalog.json 中的 skills 数组（已注入上下文）。
2. 按 ROUTING_PROTOCOL.md 的 Step 1–3 召回、选择、知会。
3. 若任一 selected skill 的 confirm 为 true，用 question 工具询问用户是否继续。
4. 用户同意后，用 skill 工具加载选中 skill。
5. 按 skill 自身说明继续完成任务。

不得选择 catalog 之外的 skill。无匹配时直接基于知识库回答。
```

然后把 `catalog.json` 的 skills 数组作为上下文提供（可通过 opencode 的 `available_skills` 机制或直接在 prompt 中附上）。

---

## 3. 与 opencode 原生 available_skills 的关系

opencode 启动时会扫描 `~/.config/opencode/skills/` 和项目 `.opencode/skills/`，生成 `available_skills` 列表注入系统提示。这本身就是一种“轻量目录”，与本系统的 catalog.json 概念一致。

两种集成方式：

### 方式 A：用 opencode 原生机制（推荐用于纯 opencode 场景）

把本仓库的 `skills/` 软链或复制到 `.opencode/skills/`，opencode 自动生成 `available_skills`。此时 catalog.json 可作为审计与跨平台交换的辅助产物。

### 方式 B：用本系统 catalog（推荐用于跨平台场景）

不依赖 opencode 原生扫描，而是用 `build_catalog.py` 生成 catalog.json，在系统提示中注入。这样可以同时接入 GitCode 仓库，且不绑定 opencode。

两种方式可共存。本系统设计为方式 B，兼容方式 A。

---

## 4. confirm 的 opencode 实现

opencode 的 `question` 工具可向用户提问并等待回答。在 Step 3 confirm 时：

```text
模型输出:
  准备使用以下 Skills：
  - npu-env-baseline：收集 Ascend NPU 环境基线（需确认：exec-shell）
  - npu-arch-capability-check：判断 NpuArch 是否支持目标算子

  其中 npu-env-baseline 需要您明确同意后才能使用。

模型调用 question 工具:
  question: "是否同意使用上述 Skills？"
  options:
    - label: "继续", description: "同意使用上述 Skills"
    - label: "调整", description: "我需要调整任务或 Skill 选择"
```

用户选“继续”后进入 Step 4；选“调整”则回到 Step 2 重新选择。

---

## 5. 加载 skill 的 opencode 实现

用 `skill` 工具加载：

```text
skill(name="npu-arch-capability-check")
```

opencode 会把对应 SKILL.md 全文注入当前对话上下文。之后模型按 skill 说明继续。

若 skill 引用 `references/` 或 `scripts/`，模型用 `read` 或 `bash` 工具按需读取。

---

## 6. 验证 checklist

opencode 适配落地后，验证以下场景：

- [ ] 用户不提 skill 名称，系统能从 catalog 召回并选择；
- [ ] 无关任务返回空 selected，不勉强匹配；
- [ ] 多步骤任务选择多个 skill 并按 order 顺序使用；
- [ ] confirm skill 在用户同意前不被加载；
- [ ] 新增 local skill（放入 `skills/`）后重跑 build_catalog 即可被发现；
- [ ] 启用 git 类型 source 后，两个 GitCode 仓库的 skill 进入 catalog。
