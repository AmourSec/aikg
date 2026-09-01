# Skills Router 维护文档

本目录实现“自动发现、选择并使用 Skills”子系统。设计见 `../AUTOMATIC_SKILL_ROUTING_DESIGN.md`（当前设计权威；原始需求文件未纳入仓库）。

本文件面向新人和 AI 维护者，覆盖新增、更新、验证和故障恢复。不依赖未写明的背景知识。

---

## 1. 整体工作原理

系统分两条相互独立的链路：

```text
构建期（维护命令可同步启用的 Git 来源；任务路由时不联网）：
sources.yaml (人维护)
      ↓
build_catalog.py 扫描所有 source 的 **/SKILL.md
      ↓
catalog.json (自动生成，轻量元数据)
      ↓
Agent 按 ROUTING_PROTOCOL.md 召回→选择→知会→加载→执行

运行期（可选，仅用户同意后联网）：
runtime/coordinator.py 协调本地候选 + 远程候选
      ↓
runtime/ascend_kg.py 访问 Ascend KG (https://ascend.wiki)
      ↓
执行目标：原生 skill 工具 / 校验路径读取 / 定界内联远程内容
```

**责任边界：**

- 构建期只负责“发现”，产出的 catalog 是本地轻量目录；Skill 数量以 `catalog.json` 当前生成为准；
- 运行期远程提供方是可选增量，见第 8 节；
- Skill 加载后的具体任务由 Skill 自身说明驱动；
- 本系统不绑定特定 Agent 平台。

---

## 2. 文件清单与维护责任

| 文件 | 维护者 | 说明 |
| --- | --- | --- |
| `config/sources.yaml` | 人 | Skill 来源真相 |
| `scripts/build_catalog.py` | 人（本系统维护者） | 通用扫描器 |
| `scripts/validate_catalog.py` | 人（本系统维护者） | 完整性校验 |
| `scripts/generate_router_context.py` | 人（本系统维护者） | 生成可注入路由上下文 |
| `scripts/test_routing.py` | 人（本系统维护者） | 回归测试 |
| `catalog.json` | 自动生成 | 不要手改 |
| `router-context.md` | 自动生成 | 不要手改,注入模型用 |
| `ROUTING_PROTOCOL.md` | 人 | 平台无关协议 |
| `opencode-adapter.md` | 人 | opencode 适配参考 |
| `runtime/contracts.py` | 人（本系统维护者） | 运行时类型与协议契约 |
| `runtime/coordinator.py` | 人（本系统维护者） | 运行时同意门与降级协调 |
| `runtime/ascend_kg.py` | 人（本系统维护者） | Ascend KG 远程提供方 |
| `runtime/ascend_kg_parsing.py` | 人（本系统维护者） | 远程候选与 SKILL.md 边界解析 |
| `runtime/http_transport.py` | 人（本系统维护者） | urllib 传输层（拒绝重定向） |
| `runtime/local_catalog.py` | 人（本系统维护者） | 本地候选加载与类型化降级 |
| `runtime/token_registry.py` | 人（本系统维护者） | 任务级不透明响应令牌 |
| `runtime/rendering.py` | 人（本系统维护者） | 远程正文定界渲染 |
| `runtime/facade.py`、`runtime/wire.py` 等 | 人（本系统维护者） | 生产门面 RouterTask 与外部契约 |
| `runtime/__main__.py`、`runtime/ndjson*.py` | 人（本系统维护者） | 可执行 NDJSON 入口（`python3 -m runtime`） |
| `tests/` | 人（本系统维护者） | 运行时单元测试（假传输层，不联网） |
| `README.md`（本文件） | 人 | 维护文档 |
| 各 source 的 `SKILL.md` | Skill 作者 | Skill 内容 |

---

## 3. 如何新增一个 Skill

### 3.1 加到本仓库（local source）

1. 在 `skills/` 下新建目录，目录名小写短横线：

   ```text
   skills/my-new-skill/
     SKILL.md
   ```

2. 写 `SKILL.md`，frontmatter 至少含 `name` 和 `description`：

   ```yaml
   ---
   name: my-new-skill
   description: Use when asked to <task>. This skill is for <scope>, especially when <triggers>. Do not use for <out-of-scope>.
   tags: [npu, env]           # 可选
   confirm: false             # 可选
   triggers: [ascend, cann]   # 可选
   ---
   ```

   写作规范见 `../docs/11-knowledge-index/skills-authoring-guide.md`。

3. 重新生成 catalog：

   ```bash
   python3 skills-router/scripts/build_catalog.py
   ```

4. 校验：

   ```bash
   python3 skills-router/scripts/validate_catalog.py
   ```

5. 确认新 skill 出现在 `catalog.json` 的 `skills` 数组中。

**不需要修改 build_catalog.py 或路由代码。** 这满足需求第 6 节“新增一个合法 Skill 后，不修改路由代码即可被发现”。

### 3.2 加到 GitCode 仓库

若 skill 在 `agent-skills` 或 `cannbot-skills` 仓库中，无需本仓库操作。只要该仓库被 sources.yaml 声明为 git source 且 `enabled: true`，build_catalog.py 会自动扫描其中的 `**/SKILL.md`。

---

## 4. 如何新增或更新一个 Skill 来源

编辑 `config/sources.yaml`，在 `sources` 下追加：

```yaml
- name: my-org-skills
  type: git
  url: https://gitcode.com/my-org/my-skills
  branch: main
  root: ""
  description: 我的组织 skills 仓库
  sync_dir: .skills-cache/my-org-skills
  enabled: true
```

或本地目录：

```yaml
- name: my-local-skills
  type: local
  root: /path/to/skills   # 相对仓库根或绝对路径
  description: 本地额外 skills
```

然后：

```bash
python3 skills-router/scripts/build_catalog.py
python3 skills-router/scripts/validate_catalog.py
```

**不需要新增专用代码分支。** 两个 GitCode 仓库（`ascend-agent-skills`、`cannbot-skills`）已用同一种 type=git 机制预留，满足需求第 9 节第 8 条。

---

## 5. 如何启用 GitCode 仓库

默认 `enabled: false`，避免网络依赖。启用步骤：

1. 编辑 `config/sources.yaml`，把对应 source 的 `enabled` 改为 `true`；
2. 确保本地有 git 命令和网络访问 gitcode.com；
3. 运行：

   ```bash
   python3 skills-router/scripts/build_catalog.py
   ```

   首次会 `git clone --depth 1` 到 `sync_dir`，后续每次 `git pull --ff-only`。

4. 校验：

   ```bash
   python3 skills-router/scripts/validate_catalog.py
   ```

`.skills-cache/` 不应提交到仓库，已加入 `.gitignore`（见第 10 节）。

---

## 6. 如何检查 Skill 是否被正确发现和选择

### 6.1 检查发现

```bash
python3 skills-router/scripts/build_catalog.py --print
```

输出 catalog 全文。确认目标 skill 在 `skills` 数组中，且 `errors` 和 `conflicts` 为空。

### 6.2 检查选择

按 `ROUTING_PROTOCOL.md` 的 Step 1–2 用自然语言任务测试。例如：

```text
任务: 请检查当前昇腾环境、判断硬件架构，并给出后续算子开发建议。
```

预期：召回 `npu-arch-capability-check`（因 description 含 ascend/npu/architecture），Step 2 选中并输出 notify。

### 6.3 检查无关任务不匹配

```text
任务: 解释 Transformer 的 self-attention 原理。
```

预期：Step 1 可能召回（关键词命中），但 Step 2 应返回空 selected，说明“未找到适合的 Skill”。

---

## 7. 如何设置 notify 或 confirm

### 7.1 单个 Skill

在 SKILL.md frontmatter：

```yaml
confirm: true    # 强制 confirm
```

或省略，由 tags 推导。

### 7.2 按类别

在 `config/sources.yaml` 的 `defaults.confirm_tags` 追加 tag：

```yaml
defaults:
  confirm_tags:
    - destructive
    - write-fs
    - network
    - exec-shell
    - my-new-category
```

任何 skill 的 tags 命中即升级为 confirm。

### 7.3 按来源

在单个 source 上：

```yaml
- name: risky-source
  type: git
  url: ...
  default_confirm: true   # 该来源所有 skill 默认 confirm
```

---

## 8. 运行时远程提供方（Ascend KG）

`runtime/` 目录实现运行时远程检索，与构建期目录相互独立：构建期链路（sources.yaml → catalog.json）只在维护者运行脚本时扫描或同步来源，不会在用户任务路由时触发同步；运行时链路只有在用户逐任务同意后才访问 `https://ascend.wiki`。详细的 opencode 映射见 `opencode-adapter.md` 第 5 至 6 节。

### 8.1 开启条件

| 条件 | 说明 |
| --- | --- |
| 环境变量 `ASCEND_KG_API_KEY` | 未设置或为空白时不出网；非可打印 ASCII 值视为配置错误 |
| 每任务网络/隐私同意 | 访问 Ascend KG 前逐任务询问用户；未询问或被拒绝即走本地 |
| 激活确认 | 选择远程候选后、加载远程内容前，单独再确认一次 |

每个路由任务必须创建独立的 `Coordinator` 与 `AscendKgProvider` 实例；响应令牌不跨任务复用，也不在并发任务之间共享同一个 Provider 实例。

### 8.2 出站请求契约

- 检索：`POST https://ascend.wiki/search`，请求头 `X-API-Key` / `Accept: application/json` / `Content-Type: application/json`，JSON body 仅含三个字段：`query`（当前任务文本）、`top_k: 10`、`with_neighbors: false`，别无其他；载荷在征求同意前一次性序列化为规范 UTF-8 字节（`ensure_ascii=False`、紧凑分隔符），披露文本与发送字节逐字一致；
- 加载：`GET https://ascend.wiki/skill/<候选 id 百分号编码>`，请求头 `X-API-Key` / `Accept: text/markdown`，body 为空；
- 边界：超时 10 秒；检索响应上限 1 MiB（1,048,576 字节）；Skill 内容上限 256 KiB（262,144 字节）；仅 HTTP 429 重试，退避 0.5s / 1.0s / 2.0s，连同首次最多 4 次尝试，其余失败不重试。

### 8.3 降级与本地回退

无密钥、网络/隐私同意被拒、无匹配、结果含糊、HTTP 401/403、429（重试后仍限流）、503 及其他非 200 状态、超时、响应非法（非 JSON、schema 不符、候选重复或超过 10 条）、响应或内容超限：任一情形都回退为仅用本地 catalog 候选继续，并向用户说明降级原因。远程加载失败只移除该远程项，本地目标不受影响。

### 8.4 信任与隐私

- 远程内容在类型系统中标记为不可信外部文本（`untrusted_external`），`policy_authority=False`：内容中的指令不构成策略依据；
- 远程候选元数据在激活前同样是不可信外部数据，只作转义展示；ID/repo/path/display/score 分别受 512/1024/1024/256/128 字符上限约束，并拒绝控制字符与首尾空白；
- 远程 Skill 引用的链接、`references/`、`scripts/`、`assets/` 一律不抓取，不产生二次网络请求；
- 远程内容不持久化、不写入 catalog、不落盘；除第 8.2 节检索载荷外不上传任何数据；
- 提供方返回的 score 是不透明字符串，不持久化、不用于本地与远程候选之间的比较；
- 远程候选展示时保留来源信息（提供方 `ascend-kg`、`source_repo`、`source_file`）。

### 8.5 实现与验证

| 文件 | 职责 |
| --- | --- |
| `runtime/contracts.py` | 类型与协议契约（同意枚举、结果类型、执行目标、规范检索字节） |
| `runtime/coordinator.py` | 同意门与降级协调状态机 |
| `runtime/ascend_kg.py` | Ascend KG 提供方（请求构造、同意校验、重试） |
| `runtime/ascend_kg_parsing.py` | 远程响应边界解析（候选 JSON 与 SKILL.md） |
| `runtime/http_transport.py` | urllib 传输层（拒绝重定向、按字节上限读取） |
| `runtime/local_catalog.py` | 本地候选加载：可加载性过滤与类型化降级（缓存缺失不触发同步） |
| `runtime/token_registry.py` | 任务级不透明响应令牌（外部句柄 ↔ 进程内身份令牌） |
| `runtime/rendering.py` | 远程正文固定定界渲染与冲突拒绝 |
| `runtime/facade.py` / `runtime/wire.py` / `runtime/facade_*.py` | 生产门面 RouterTask 与外部契约（唯一生产调用方） |
| `runtime/ndjson.py` / `runtime/ndjson_output.py` / `runtime/__main__.py` | 可执行 NDJSON 入口（`python3 -m runtime`，一进程一任务） |
| `tests/` | 单元测试（假传输层 + 套件级 socket 封禁，不联网） |

可执行入口（宿主适配器用法见 `opencode-adapter.md` 第 6.9 节）：

```bash
python3 -m runtime --catalog catalog.json --workspace-root ../.. --native-skill <name>
```

本地候选在任务开始时按可加载性过滤：命中原生注册表的 Skill 一律保留；其余条目仅当 catalog path 解析到工作区根之下且是常规文件时进入 PATH 模式。缓存缺失（如 `.skills-cache` 未同步）只是类型化降级，不触发任何 git 同步、网络请求或安装动作。

在 `skills-router/` 目录下运行单元测试：

```bash
python3 -m unittest discover -s tests -t .
```

与上游 kg-tools 的关系：仅作设计参照。上游仓库为 [`agent0/kg-tools`](https://gitcode.com/agent0/kg-tools)，固定参照 commit 为 [`5568d8eedc70eebf155cd4e2728aee93ea02962d`](https://gitcode.com/agent0/kg-tools/commit/5568d8eedc70eebf155cd4e2728aee93ea02962d)；kg-tools 编排引擎未集成，未 vendor 任何上游代码。本文档不声称已对线上 API 做过实测。

---

## 9. 故障恢复

### 9.1 build_catalog 报错

| 错误 | 原因 | 处理 |
| --- | --- | --- |
| `sources.yaml not found` | 配置文件被删 | 从 git 恢复 `config/sources.yaml` |
| `yaml parse error` | frontmatter 语法错 | 检查对应 SKILL.md 的 `---` 段 |
| `missing 'name'` | frontmatter 缺字段 | 补 name，与目录名一致 |
| `description too short` | description < 20 字符 | 重写 description，写清触发条件 |
| `root directory missing` | source 根目录不存在 | 检查 sources.yaml 的 root 或 sync_dir |
| `git clone failed` | 网络或 URL 问题 | 检查 url、branch、网络；可临时 `enabled: false` |
| `conflict: skill 'X' defined in N paths` | 跨来源同名 | 改其中一个的 name，或删除重复 |

错误不会中断整体扫描，会收集到 catalog.json 的 `errors` / `conflicts` 数组。修复后重跑即可。

### 9.2 catalog.json 缺失或损坏

```bash
python3 skills-router/scripts/build_catalog.py
```

重新生成。catalog.json 是自动产物，删了再生成即可。

### 9.3 元数据漂变

Skill 作者改了 SKILL.md 的 name/description 但没重跑 build_catalog，导致 catalog 与文件不一致。`validate_catalog.py` 会检测 name drift：

```bash
python3 skills-router/scripts/validate_catalog.py
```

修复方式：重跑 build_catalog。

### 9.4 git source 拉取失败

临时把 `enabled` 改为 `false`，build_catalog 会跳过该 source 继续生成其余 catalog。修复网络后改回 `true`。

### 9.5 远程提供方不可用或被拒

运行时远程提供方的所有失败都不会中断本地路由，系统回退为仅用本地 catalog 候选，并向用户说明降级原因（见第 8.3 节）。

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 始终本地路由，无网络请求 | `ASCEND_KG_API_KEY` 未设置或为空白 | 按需设置环境变量 |
| 降级原因为 `configuration` | 密钥无效或无权限（HTTP 401/403） | 检查 API 密钥 |
| 降级原因为 `rate_limited` | HTTP 429 重试后仍限流 | 稍后再试 |
| 降级原因为 `service` / `timeout` | HTTP 503、其他非 200 状态或超时 | 检查网络与上游服务状态 |
| 降级原因为 `invalid_json` / `invalid_schema` / `oversized` | 响应非法或超过字节上限 | 属上游契约问题，保持本地回退即可 |

---

## 10. .gitignore 配置

仓库根 `.gitignore` 应包含：

```text
.skills-cache/
```

避免把 git source 的克隆内容提交进来。`.skills-cache/` 是本地缓存，可随时删除重建。

---

## 11. 完整验证命令

```bash
# 1. 同步来源并生成 catalog（git 来源需本地 .skills-cache 同步后才完整）
python3 skills-router/scripts/build_catalog.py

# 2. 校验完整性（同上：缓存缺失时 catalog 中缓存来源条目会报 missing）
python3 skills-router/scripts/validate_catalog.py

# 3. 生成可注入的路由上下文
python3 skills-router/scripts/generate_router_context.py

# 4. 跑回归测试（依赖 catalog 与缓存文件存在）
python3 skills-router/scripts/test_routing.py

# 5. 查看 catalog
python3 skills-router/scripts/build_catalog.py --print

# 6. 跑运行时远程提供方单元测试（假传输层 + socket 封禁，不联网，不依赖 .skills-cache）
cd skills-router && python3 -m unittest discover -s tests -t .

# 7. 可执行入口自检（无网络即可成功）
cd skills-router && python3 -m runtime --help
```

预期结果：

- `build_catalog.py` 报告 N skills、0 conflicts(可接受远程仓库内部的 conflicts/errors);
- `validate_catalog.py` 输出 `OK: N skills, 0 problems`（`.skills-cache` 缺失时会列出 missing 文件，属预期降级而非错误）;
- `generate_router_context.py` 生成 `router-context.md`;
- `test_routing.py` 6 项全部 PASS（需 `.skills-cache` 完整；缓存缺失时 `catalog_valid` / `load_skill` 会失败，可先运行 build_catalog.py 同步来源）;
- `unittest discover` 全部通过;
- `python3 -m runtime --help` 正常打印用法。

---

## 12. AI 修改本系统时必须遵守的步骤

当 AI 协助维护本系统时，必须：

1. **先读** `../AUTOMATIC_SKILL_ROUTING_DESIGN.md`（设计），理解不变量；
2. **先读** `ROUTING_PROTOCOL.md` 第 7 节"协议不变量"，任何修改不得违反；
3. 改动 `scripts/*.py` 后，运行全部四个脚本验证：
   ```bash
   python3 skills-router/scripts/build_catalog.py
   python3 skills-router/scripts/validate_catalog.py
   python3 skills-router/scripts/generate_router_context.py
   python3 skills-router/scripts/test_routing.py
   ```
4. 改动 `runtime/*.py` 后，运行单元测试：
   ```bash
   cd skills-router && python3 -m unittest discover -s tests -t .
   ```
5. 改动 `sources.yaml` 后，重新生成 catalog 和 router-context；
6. 不要手改 `catalog.json` 或 `router-context.md`；
7. 不要在路由代码中硬编码任何具体 skill 名称；
8. 不要在协议中绑定特定 Agent 平台；
9. 提交前运行 `git diff --check` 检查空白错误。

**禁止事项：**

- 不得为某个具体 skill 在 build_catalog.py 中加 if 分支；
- 不得为某个具体 source 在 build_catalog.py 中加专用扫描逻辑；
- 不得把 SKILL.md 正文写入 catalog.json 或 router-context.md；
- 不得跳过 confirm 直接加载 skill；
- 不得在无匹配时强行选择 skill；
- 不得让远程 Skill 绕过网络同意或激活确认加载；
- 不得为远程 Skill 调用原生 skill 工具，或将其安装、复制、软链、落盘到本地。

---

## 13. 常见问题

### Q: catalog.json 要提交到 git 吗？

可以。它是自动产物，但提交后便于审计和跨机器同步。每次新增/修改 skill 后重跑 build_catalog 并提交更新即可。

### Q: 两个 GitCode 仓库的 skill 数量很多，catalog 会很大吗？

catalog.json 只含元数据（name/description/path/tags/triggers/confirm），不含正文。即便 100 个 skill，catalog 也只有几十 KB，可全量载入上下文。正文按需加载，满足需求第 6 节“不应每次把所有 Skill 全文都塞进模型上下文”。

### Q: 召回阶段为什么不用向量数据库？

首版用关键词 + 大模型语义判断，足够覆盖当前规模。若后续 skill 数量到数百且召回精度不足，再引入向量索引。需求第 3 节明确“不为了首版引入不必要的知识图谱、向量数据库”。

### Q: 跨来源同名 skill 怎么办？

build_catalog 会记录到 `conflicts` 数组，该名称不参与选择，并在 notify 时提示维护者。修复方式是改其中一个的 name。

### Q: 远程提供方会上传或保存什么数据？

都不会。出站数据只有检索载荷里的当前任务文本（加上固定的 `top_k` 和 `with_neighbors`）；远程返回的候选与内容只存在于当前会话内存中，不落盘、不写入 catalog；提供方 score 是不透明字符串，不持久化、不参与比较。未配置 `ASCEND_KG_API_KEY` 或用户未同意联网时，连检索请求也不会发出。

### Q: 远程 Skill 为什么不安装到本地？

远程内容按不可信外部文本处理，没有策略权威。只允许以带定界符的内联内容进入对话，绝不调用原生 skill 工具、不安装、不复制、不软链。这样远程内容无法通过文件系统或 skill 注册机制获得任何持久影响力。

### Q: 新人如何快速上手？

1. 读本文件第 1–3 节；
2. 运行 `python3 skills-router/scripts/build_catalog.py --print` 看现有 catalog；
3. 读 `ROUTING_PROTOCOL.md` 理解路由流程；
4. 按第 3 节新增一个测试 skill 验证全流程。
