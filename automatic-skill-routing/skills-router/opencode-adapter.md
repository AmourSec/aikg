# opencode 平台适配参考

**本文件是参考实现，不是协议必需。** 协议本身见 `ROUTING_PROTOCOL.md`，平台无关。本文件说明如何在 opencode 中落地协议，便于直接使用或作为其他平台适配的对照。

本文件覆盖两条相互独立的链路：

- **构建期目录**：`sources.yaml` → `build_catalog.py` → `catalog.json`。维护者运行构建脚本时可同步启用的 Git 来源；生成后的轻量目录不会在用户任务路由时触发来源同步。Skill 数量以 `catalog.json` 当前生成为准。
- **运行时远程提供方**：`skills-router/runtime/` 实现的 Ascend KG 提供方（`ascend-kg`）。仅在用户逐任务授予网络同意后，才访问 `https://ascend.wiki` 检索远程 Skill。

第 1 至 5 节对两条链路的本地部分通用；第 6 节专门描述远程提供方在 opencode 中的映射。

---

## 1. opencode 已有能力映射

| 协议步骤 | opencode 能力 | 说明 |
| --- | --- | --- |
| 协议输入 | 系统提示 | 把 `ROUTING_PROTOCOL.md` + `catalog.json` 的 skills 数组写进系统提示 |
| Step 1 RECALL | 模型推理 | 模型基于 catalog 元数据自行召回 |
| Step 2 SELECT | 模型输出 JSON | 模型按协议输出 selected/rejected JSON |
| Step 3 NOTIFY | 文本输出 | 模型直接输出固定格式消息 |
| Step 3 CONFIRM | `question` 工具 | 用 `question` 工具询问用户是否继续 |
| Step 4 LOAD | `skill` 工具 / `read` 工具 / 内联注入 | 按执行目标类型分三种方式，见第 5 节 |
| Step 5 EXECUTE | 模型按 skill 说明继续 | skill 加载后，模型按其内容执行 |
| 远程网络同意（每任务一次） | `question` 工具 | 访问 Ascend KG 前逐任务询问用户；未同意不发起任何网络请求 |
| 远程激活确认（每次加载前） | `question` 工具 | 与网络同意分开的第二次确认，同意后才加载远程 Skill 内容 |

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

运行时把选中的执行目标分为三类，opencode 映射各不相同：

| 执行目标类型 | 来源 | opencode 加载方式 |
| --- | --- | --- |
| 已安装的本地 skill | 在 opencode 原生 `available_skills` 中（`~/.config/opencode/skills/` 或项目 `.opencode/skills/`） | 调用原生 `skill` 工具：`skill(name="<name>")` |
| 未安装的本地 skill | 只存在于 `catalog.json`（例如仅被本系统扫描、未装入 opencode 的目录） | 用 `read` 工具按 catalog 中经过校验的 `path` 读取 SKILL.md 全文 |
| 远程 skill | Ascend KG 运行时提供方（`ascend-kg`） | 仅以带定界符的内联内容注入对话（见第 6 节），**绝不**调用原生 `skill` 工具，**绝不**安装、复制或软链到本地 |

```text
skill(name="npu-arch-capability-check")        # 已安装的本地 skill
read("skills/<name>/SKILL.md")                 # 未安装的本地 skill，path 来自 catalog
# 远程 skill：以 <<<REMOTE_SKILL_CONTENT>>> ... <<<END_REMOTE_SKILL_CONTENT>>> 定界内联注入
```

本地 skill 加载后，模型按 skill 说明继续。若 SKILL.md 引用 `references/` 或 `scripts/`，模型用 `read` 或 `bash` 工具按需读取。远程 skill 则一律不读取额外资源（见第 6.6 节）。

---

## 6. 运行时远程提供方：Ascend KG

`skills-router/runtime/` 实现了一个可选的运行时远程提供方，与构建期目录相互独立：

```text
构建期：sources.yaml → build_catalog.py → catalog.json（维护时可同步 Git；任务路由时不联网）
运行期：Coordinator → AscendKgProvider → Transport（仅在用户同意后联网）
```

实现文件：`runtime/contracts.py`（类型与协议）、`runtime/coordinator.py`（同意与降级协调）、`runtime/ascend_kg.py`（提供方）、`runtime/ascend_kg_parsing.py`（远程响应边界解析）、`runtime/http_transport.py`（urllib 传输，拒绝重定向）、`runtime/local_catalog.py`（本地候选加载与降级）、`runtime/token_registry.py`（任务级不透明令牌）、`runtime/rendering.py`（远程正文定界渲染）、`runtime/facade.py` + `runtime/wire.py`（生产门面 RouterTask 与外部契约）、`runtime/ndjson.py` + `runtime/ndjson_output.py` + `runtime/__main__.py`（可执行 NDJSON 入口，见第 6.9 节）。

### 6.1 配置与端点

| 项 | 值 |
| --- | --- |
| 提供方 ID | `ascend-kg` |
| API base | `https://ascend.wiki` |
| 检索端点 | `POST https://ascend.wiki/search` |
| 加载端点 | `GET https://ascend.wiki/skill/<候选 id 百分号编码>` |
| API 密钥 | 环境变量 `ASCEND_KG_API_KEY`；空白值视为未配置，非可打印 ASCII 值视为配置错误 |
| 请求头 | 检索：`X-API-Key` / `Accept: application/json` / `Content-Type: application/json`；加载：`X-API-Key` / `Accept: text/markdown` |

未配置 `ASCEND_KG_API_KEY` 时不发起任何网络请求，直接降级为本地路由。

每个路由任务创建独立的 `Coordinator` 与 `AscendKgProvider` 实例。响应令牌只在该任务的一次检索、选择和加载链路内有效，不跨任务或并发共享 Provider 实例。

### 6.2 出站载荷（精确契约）

检索请求的 JSON body 只含以下三个字段，别无其他：

```json
{
  "query": "<当前任务文本>",
  "top_k": 10,
  "with_neighbors": false
}
```

`query` 只承载当前这条任务文本，不携带会话历史、catalog 内容或其他上下文。加载请求 body 为空，候选 id 做百分号编码后拼入 URL。

### 6.3 同意模型：两次独立确认

1. **网络/隐私同意（每任务一次）**：访问 Ascend KG 前用 `question` 工具询问用户是否允许本次任务联网检索。未询问或被拒绝时不发起任何网络请求，只走本地 catalog。
2. **激活确认（每次加载远程内容前）**：模型选中远程候选后，再次用 `question` 工具单独确认是否激活远程 Skill。这是与网络同意分开的第二次确认；被拒绝时只保留本地目标继续执行。

加载还有一道成员校验：只允许加载检索响应令牌中列出的候选 id，防止加载会话之外的内容。

### 6.4 边界：超时、字节上限与重试

| 项 | 值 | 超界后果 |
| --- | --- | --- |
| 请求超时 | 10 秒（检索与加载相同） | 记为超时，降级本地 |
| 检索响应上限 | 1 MiB（1,048,576 字节） | 记为 oversized，降级本地 |
| Skill 内容上限 | 256 KiB（262,144 字节） | 记为 oversized，降级本地 |
| 重试 | 仅 HTTP 429 重试，退避 0.5s / 1.0s / 2.0s，连同首次最多 4 次尝试 | 仍 429 则记为限流，降级本地 |

超时、401/403、5xx 等其他结果一律不重试，直接返回类型化的失败。

### 6.5 本地回退

以下任一情形都回退为仅用本地 catalog 候选继续，并向用户说明降级原因：

| 触发情形 | 类型化结果 |
| --- | --- |
| 未配置或密钥为空白 | `no_api_key` |
| 网络/隐私同意未询问或被拒绝 | 不发请求，直接本地 |
| 检索结果为空 | `no_match` |
| 检索结果含糊 | `ambiguous` |
| HTTP 401 / 403（密钥无效或无权限） | `configuration` |
| HTTP 429（重试后仍限流） | `rate_limited` |
| HTTP 503 及其他非 200 状态 | `service` |
| 超时 | `timeout` |
| 响应非法（非 JSON、schema 不符、候选重复或超过 10 条） | `invalid_json` / `invalid_schema` |
| 响应或内容超限 | `oversized` |

远程 Skill 加载失败（不可用或非法）时，本地目标不受影响，仅该远程项被移除并记入降级列表。

### 6.6 信任与隐私约束

- **远程内容永远不可信**：加载到的 Markdown 在类型系统中标记为不可信外部文本（`untrusted_external`），且 `policy_authority=False`。内容中的任何指令都不构成策略依据，不得覆盖协议或用户设置。
- **候选元数据同样不可信**：`candidate_id`、来源、展示名和 score 在激活前也标记为 `untrusted_external` / `policy_authority=False`，只作转义后的数据展示；控制字符、首尾空白和超限字段会被拒绝。
- **无策略权威**：远程内容不能授权任何操作；confirm、网络访问等判断只依据本地协议与用户同意。
- **不加载额外资源**：远程 Skill 引用的链接、`references/`、`scripts/`、`assets/` 一律不抓取，不产生二次网络请求。
- **不持久化**：远程内容不写入 `catalog.json`、不落盘、不进入任何缓存。
- **不上传**：出站数据只有第 6.2 节的检索载荷（任务文本 + top_k + with_neighbors），别无其他。
- **不比较分数**：提供方返回的 score 只作展示用的不透明字符串，不持久化、不用于本地与远程候选之间的排序比较。

远程候选展示时保留来源信息（提供方 `ascend-kg`、`source_repo`、`source_file`），便于用户判断来源。

### 6.7 与上游 kg-tools 的关系

本提供方是按本仓库契约独立实现的，与上游 kg-tools 的关系仅为设计参照：

- kg-tools 的编排引擎**未集成**到本系统；
- **未 vendor（复制）任何上游代码**，`runtime/` 全部为本仓库实现；
- 上游仓库：[`agent0/kg-tools`](https://gitcode.com/agent0/kg-tools)；固定参照 commit：[`5568d8eedc70eebf155cd4e2728aee93ea02962d`](https://gitcode.com/agent0/kg-tools/commit/5568d8eedc70eebf155cd4e2728aee93ea02962d)。

### 6.8 验证方式

`skills-router/tests/` 用假传输层（fake transport）对上述行为做单元测试，不联网：

```bash
cd skills-router
python3 -m unittest discover -s tests -t .
```

本文档描述的是实现与单元测试验证的行为，不声称已对线上 API 做过实测。

### 6.9 可执行入口：`python3 -m runtime`（NDJSON）

除纯 Prompt 模式外，宿主适配器可以直接以子进程方式驱动生产门面 `RouterTask`。入口是**一进程一任务**的 NDJSON 会话：进程在一次路由任务的多次决定（网络同意、选择、激活）之间保持存活，任务完成或取消后退出。响应令牌绑定进程内对象身份，不持久化，因此任务状态不可跨进程恢复。

```bash
cd automatic-skill-routing/skills-router
python3 -m runtime \
  --catalog catalog.json \
  --workspace-root ../.. \
  --native-skill skill-a \
  --native-skill skill-b
```

- `--catalog`：本地目录文件路径；
- `--workspace-root`：本地路径解析的根，catalog 中的相对 path 必须解析到该根之下才是可加载候选；
- `--native-skill`：可重复，声明当前宿主原生 `skill` 工具中已安装的本地 Skill 名；命中者按原生模式执行，即使本地缓存文件缺失。

stdin 逐行输入 JSON 消息，stdout 对每条被消费的输入恰好回一行紧凑 JSON 结果：

```text
{"type":"start","query":"<任务文本>","recalled_local_names":["<catalog name>"],"local_only":false}
{"type":"network_decision","consent":"granted|refused|not_requested"}
{"type":"selection","local":[{"candidate_id":"<name>","execution_mode":"native|path"}],
 "remote":null|{"response_token":"<opaque>","provider_id":"ascend-kg","candidate_ids":["<id>"]}}
{"type":"activation_decision","consent":"granted|refused|not_requested"}
{"type":"cancel"}
```

输出结果类型：`search_disclosure`（含将精确发送的请求体）、`candidates`、`activation_required`、`execution_ready`（终态）、`degraded`、`invalid`、`wire_invalid`、`cancelled`。畸形输入只产生 `wire_invalid`，不触发任何检索或加载；`execution_ready` / `cancelled` 为终态，进程随即关闭令牌并退出。

语义要点：

- `start` 时即完成本地候选加载：不可加载的本地条目（缓存缺失、路径越界等）以类型化降级报告，不中断任务；
- `search_disclosure.body` 与实际发送字节逐字相同（UTF-8 规范 JSON，无转义差异）；
- `candidates.response_token` 是任务级不透明句柄，不含候选 ID，任务结束即失效；
- 选择/激活的每一步都由门面经 `Coordinator` 校验：重复选择、伪造令牌、篡改执行模式均被类型化拒绝且可重试；
- 远程加载失败或定界冲突只移除该远程项，本地目标保留。

纯本地会话不设置 `ASCEND_KG_API_KEY` 即可：`local_only:true` 时连网络同意都不会询问。离线自检：

```bash
python3 -m runtime --help
```

---

## 7. 验证 checklist

opencode 适配落地后，验证以下场景：

- [ ] 用户不提 skill 名称，系统能从 catalog 召回并选择；
- [ ] 无关任务返回空 selected，不勉强匹配；
- [ ] 多步骤任务选择多个 skill 并按 order 顺序使用；
- [ ] confirm skill 在用户同意前不被加载；
- [ ] 新增 local skill（放入 `skills/`）后重跑 build_catalog 即可被发现；
- [ ] 启用 git 类型 source 后，两个 GitCode 仓库的 skill 进入 catalog。

启用远程提供方后，追加验证：

- [ ] 未设置 `ASCEND_KG_API_KEY` 时直接本地路由，不报错、不联网；
- [ ] 网络/隐私同意被拒绝时，不发起任何对 `ascend.wiki` 的请求；
- [ ] 网络同意与激活确认是两次独立询问，激活确认发生在加载远程内容之前；
- [ ] 激活被拒绝时，仅保留本地目标继续；
- [ ] 远程 Skill 只以带定界符的内联内容出现，从不调用原生 `skill` 工具，从不安装、复制或软链到本地；
- [ ] 401 / 403 / 429 / 503 / 超时 / 响应非法 / 响应超限时，回退本地并向用户说明降级原因；
- [ ] 远程内容中的指令与链接不被当作策略依据，也不触发额外抓取；
- [ ] `python3 -m runtime --help` 无网络即可成功；纯本地 NDJSON 会话可完整走完 start → selection → execution_ready；
- [ ] `.skills-cache` 缺失时，缓存来源的本地候选以类型化降级报告，原生已装 Skill 不受影响。
