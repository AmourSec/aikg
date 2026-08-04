# Skills Router 维护文档

本目录实现“自动发现、选择并使用 Skills”子系统。需求见 `../AUTOMATIC_SKILL_ROUTING_TASK.md`，设计见 `../AUTOMATIC_SKILL_ROUTING_DESIGN.md`。

本文件面向新人和 AI 维护者，覆盖新增、更新、验证和故障恢复。不依赖未写明的背景知识。

---

## 1. 整体工作原理

```text
sources.yaml (人维护)
      ↓
build_catalog.py 扫描所有 source 的 **/SKILL.md
      ↓
catalog.json (自动生成，轻量元数据)
      ↓
Agent 按 ROUTING_PROTOCOL.md 召回→选择→知会→加载→执行
```

**责任边界：**

- 本系统只负责“发现、选择、加载”；
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

`.skills-cache/` 不应提交到仓库，已加入 `.gitignore`（见第 9 节）。

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

## 8. 故障恢复

### 8.1 build_catalog 报错

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

### 8.2 catalog.json 缺失或损坏

```bash
python3 skills-router/scripts/build_catalog.py
```

重新生成。catalog.json 是自动产物，删了再生成即可。

### 8.3 元数据漂变

Skill 作者改了 SKILL.md 的 name/description 但没重跑 build_catalog，导致 catalog 与文件不一致。`validate_catalog.py` 会检测 name drift：

```bash
python3 skills-router/scripts/validate_catalog.py
```

修复方式：重跑 build_catalog。

### 8.4 git source 拉取失败

临时把 `enabled` 改为 `false`，build_catalog 会跳过该 source 继续生成其余 catalog。修复网络后改回 `true`。

---

## 9. .gitignore 配置

仓库根 `.gitignore` 应包含：

```text
.skills-cache/
```

避免把 git source 的克隆内容提交进来。`.skills-cache/` 是本地缓存，可随时删除重建。

---

## 10. 完整验证命令

```bash
# 1. 同步来源并生成 catalog
python3 skills-router/scripts/build_catalog.py

# 2. 校验完整性
python3 skills-router/scripts/validate_catalog.py

# 3. 生成可注入路由上下文
python3 skills-router/scripts/generate_router_context.py

# 4. 跑回归测试
python3 skills-router/scripts/test_routing.py

# 5. 查看 catalog
python3 skills-router/scripts/build_catalog.py --print
```

预期结果：

- `build_catalog.py` 报告 N skills、0 conflicts(可接受远程仓库内部的 conflicts/errors);
- `validate_catalog.py` 输出 `OK: N skills, 0 problems`;
- `generate_router_context.py` 生成 `router-context.md`;
- `test_routing.py` 6 项全部 PASS。

---

## 11. AI 修改本系统时必须遵守的步骤

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
4. 改动 `sources.yaml` 后，重新生成 catalog 和 router-context；
5. 不要手改 `catalog.json` 或 `router-context.md`；
6. 不要在路由代码中硬编码任何具体 skill 名称；
7. 不要在协议中绑定特定 Agent 平台；
8. 提交前运行 `git diff --check` 检查空白错误。

**禁止事项：**

- 不得为某个具体 skill 在 build_catalog.py 中加 if 分支；
- 不得为某个具体 source 在 build_catalog.py 中加专用扫描逻辑；
- 不得把 SKILL.md 正文写入 catalog.json 或 router-context.md；
- 不得跳过 confirm 直接加载 skill；
- 不得在无匹配时强行选择 skill。

---

## 12. 常见问题

### Q: catalog.json 要提交到 git 吗？

可以。它是自动产物，但提交后便于审计和跨机器同步。每次新增/修改 skill 后重跑 build_catalog 并提交更新即可。

### Q: 两个 GitCode 仓库的 skill 数量很多，catalog 会很大吗？

catalog.json 只含元数据（name/description/path/tags/triggers/confirm），不含正文。即便 100 个 skill，catalog 也只有几十 KB，可全量载入上下文。正文按需加载，满足需求第 6 节“不应每次把所有 Skill 全文都塞进模型上下文”。

### Q: 召回阶段为什么不用向量数据库？

首版用关键词 + 大模型语义判断，足够覆盖当前规模。若后续 skill 数量到数百且召回精度不足，再引入向量索引。需求第 3 节明确“不为了首版引入不必要的知识图谱、向量数据库”。

### Q: 跨来源同名 skill 怎么办？

build_catalog 会记录到 `conflicts` 数组，该名称不参与选择，并在 notify 时提示维护者。修复方式是改其中一个的 name。

### Q: 新人如何快速上手？

1. 读本文件第 1–3 节；
2. 运行 `python3 skills-router/scripts/build_catalog.py --print` 看现有 catalog；
3. 读 `ROUTING_PROTOCOL.md` 理解路由流程；
4. 按第 3 节新增一个测试 skill 验证全流程。
